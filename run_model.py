import argparse
import logging
import os
import random
import time
import json
from datetime import datetime
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.nn import CosineSimilarity
from scipy.stats import spearmanr

from torch.nn import CrossEntropyLoss

from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
)
from transformers.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME, CONFIG_NAME
)

from tqdm import tqdm
from deepmistake.utils import (
    models, configs, DataProcessor
)
from deepmistake.utils import get_dataloader_and_tensors
from deepmistake.deepmistake import DeepMistakeWiC
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report, accuracy_score
)
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("__scores__")
test_logger = logging.getLogger("__test__")


def predict(
        model, eval_dataloader, output_dir,
        eval_fearures, args,
        cur_train_mean_loss=None,
        logger=None, compute_metrics=True,
        eval_script_path='../MeasEval/eval/measeval-eval.py',
        only_parts='',
        dump_feature=True
    ):

    if os.path.exists(output_dir):
        os.system(f'rm -r {output_dir}/*')
    else:
        os.makedirs(output_dir, exist_ok=True)

    only_parts = [part for part in only_parts.split('+') if part]
    model.eval()
    syns = sorted(model.local_config['syns'])
    device = torch.device('cuda') if model.local_config['use_cuda'] else torch.device('cpu')

    eval_dataloader = tqdm(eval_dataloader, total=len(eval_dataloader), desc='validation ... ')
    metrics, syns_preds, syns_scores_res = run_inference(eval_dataloader, model, device, compute_metrics, dump_feature,
                                                         output_dir)

    predictions = defaultdict(lambda: defaultdict(list))
    golds = defaultdict(lambda: defaultdict(list))
    scores = defaultdict(lambda: defaultdict(list))
    gold_scores = defaultdict(lambda: defaultdict(list))
    lemmas = defaultdict(lambda: defaultdict(list))

    syn_ids_to_label = {0: 'F', 1: 'T'}
    for ex_id, (ex_feature, ex_syn_preds, ex_scores) in enumerate(zip(eval_fearures, syns_preds, syns_scores_res)):
        example = ex_feature.example
        docId = example.docId
        posInDoc = int(docId.split('.')[-1])
        docId = '.'.join(docId.split('.')[:-1])
        syn_pred = syn_ids_to_label[ex_syn_preds.item()]
        predictions[docId][posInDoc].append(syn_pred)
        golds[docId][posInDoc].append(example.label)
        scores[docId][posInDoc].append(ex_scores)
        gold_scores[docId][posInDoc].append(example.score)
        lemmas[docId][posInDoc].append((example.lemma, example.grp))


    print(f'saving predictions for: {only_parts}')
    for docId, doc_preds in predictions.items():
        doc_scores = scores[docId]
        if len(only_parts) > 0 and all([f'{docId.split(".")[1]}.score' not in part for part in only_parts]):
            continue
        print(f'saving predictions for part: {docId}')
        prediction = [{'id': f'{docId}.{pos}','tag': 'F' if 'F' in doc_preds[pos] else 'T'} for pos in sorted(doc_preds)]
        prediction_file = os.path.join(output_dir, docId)
        json.dump(prediction, open(prediction_file, 'w'))
        prediction = [{'id': f'{docId}.{pos}','score': [str(x) for x in doc_scores[pos]]} for pos in sorted(doc_preds)]
        prediction_file = os.path.join(output_dir, f'{docId}.scores')
        json.dump(prediction, open(prediction_file, 'w'))

    if compute_metrics:
        mean_non_english = []
        for docId, doc_preds in predictions.items():
            if 'scd' in docId:
                doc_golds = gold_scores[docId]
                doc_lemmas = lemmas[docId]
                doc_scores = scores[docId]

                keys = sorted(list(doc_golds.keys()))
                # print(doc_lemmas)
                unique_lemmas = sorted(set([doc_lemmas[key][0][0] for key in keys if doc_lemmas[key][0][1] == 'COMPARE']))
                y_true, y_pred = [], []
                y_sent_true, y_sent_pred = [], []
                for unique_lemma in unique_lemmas:
                    unique_lemma_keys = [key for key in keys if doc_lemmas[key][0][0] == unique_lemma and doc_lemmas[key][0][1] == 'COMPARE']
                    unique_word_scores_pred = [np.array(doc_scores[key]).mean() for key in unique_lemma_keys]
                    unique_word_scores_true = [doc_golds[key][0] for key in unique_lemma_keys]
                    y_true.append(np.array(unique_word_scores_true).mean())
                    y_pred.append(np.array(unique_word_scores_pred).mean())
                    y_sent_true.extend(unique_word_scores_true)
                    y_sent_pred.extend(unique_word_scores_pred)
                # print(y_true, y_pred)
                # metrics[f'spearman.{docId}.score'], _ = spearmanr(y_true, y_pred)
                # metrics[f'spearman.{docId}.pairwise'], _ = spearmanr(y_sent_true, y_sent_pred)
                metrics[f'spearman.{docId}.wordwise.score'], _ = spearmanr(y_true, y_pred)
                metrics[f'spearman.{docId}.score'], _ = spearmanr(y_sent_true, y_sent_pred)
                doc_golds = golds[docId]
                keys = list(doc_golds.keys())
                doc_golds = [doc_golds[key][0] for key in keys]
                doc_preds = ['F' if 'F' in doc_preds[key] else 'T' for key in keys]
                metrics[f'{docId}.accuracy'] = accuracy_score(doc_golds, doc_preds)
            else:
                doc_golds = golds[docId]
                keys = list(doc_golds.keys())
                doc_golds = [doc_golds[key][0] for key in keys]
                doc_preds = ['F' if 'F' in doc_preds[key] else 'T' for key in keys]
                metrics[f'accuracy.{docId}.score'] = accuracy_score(doc_golds, doc_preds)
                if 'en-en' not in docId:
                    mean_non_english.append(metrics[f'accuracy.{docId}.score'])
        if mean_non_english:
            metrics[f'accuracy.{docId.split(".")[0]}.nen-nen.score'] = sum(mean_non_english) / len(mean_non_english)


        if cur_train_mean_loss is not None:
            metrics.update(cur_train_mean_loss)
    else:
        metrics = {}

    model.train()

    return metrics


def run_inference(eval_dataloader, model, device, compute_metrics, dump_feature, output_dir):
    metrics = defaultdict(float)
    nb_eval_steps = 0
    syns_preds = []
    for batch_id, batch in enumerate(eval_dataloader):
        batch = tuple([elem.to(device) for elem in batch])

        input_ids, input_mask, token_type_ids, b_syn_labels, b_positions = batch
        with torch.no_grad():
            loss, syn_logits, syn_features = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=input_mask,
                input_labels={
                    'syn_labels': b_syn_labels,
                    'positions': b_positions
                },
                return_features=True
            )

        if dump_feature:
            syn_features = syn_features.detach().cpu().numpy()
            np.save(os.path.join(output_dir, f'features_batch_{batch_id}.npy'), syn_features)
            np.save(os.path.join(output_dir, f'labels_batch_{batch_id}.npy'), b_syn_labels.detach().cpu().numpy())

        if compute_metrics:
            for key, value in loss.items():
                metrics[f'eval_{key}_loss'] = (metrics[f'eval_{key}_loss'] * nb_eval_steps +
                                               value.mean().item()) / (nb_eval_steps + 1)

        nb_eval_steps += 1
        if model.local_config['loss'] != 'cosine_similarity':
            syns_preds.append(syn_logits.detach().cpu().numpy())
        else:
            syns_preds.append(CosineSimilarity()(syn_logits[0], syn_logits[1]).detach().cpu().numpy())
    syns_scores = np.concatenate(syns_preds, axis=0)  # n_examples x 2 or n_examples
    # scores for positive class; this replaces the previous code in processing each score individually:
    # if model.local_config['loss'] == 'cosine_similarity':
    #     scores[docId][posInDoc].append(ex_scores)
    # elif len(ex_scores) > 1:
    #     scores[docId][posInDoc].append(softmax(ex_scores)[-1])
    # else:
    #     scores[docId][posInDoc].append(ex_scores[0])
    # TODO: TEST for different types of heads!
    if model.local_config['loss'] == 'cosine_similarity':
        syns_scores_res = syns_scores
    elif syns_scores.ndim > 1 and syns_scores.shape[-1] > 1:
        syns_scores_res = softmax(syns_scores, axis=-1)[:, -1]
    else:
        syns_scores_res = syns_scores[:, 0]
    # Create predictions from scores
    if syns_scores.shape[-1] != 1 and model.local_config['loss'] != 'cosine_similarity':
        syns_preds = np.argmax(syns_scores, axis=1)  # n_examples
    elif model.local_config['loss'] == 'cosine_similarity':
        syns_preds = np.zeros(syns_scores.shape, dtype=int)
        syns_preds[syns_scores >= 0.5] = 1
    else:
        syns_preds = np.zeros(syns_scores.shape, dtype=int)
        if model.local_config['train_scd']:
            syns_preds[syns_scores >= 3.0] = 1
        else:
            syns_preds[syns_scores > 0.5] = 1
    return metrics, syns_preds, syns_scores_res


def freeze(model, trainable_params, epoch):
    for n,t in model.named_parameters():
        t.requires_grad = any(((s.split(':')[0] in n) and (':' not in s or epoch>=int(s.split(':')[1])) for s in trainable_params.split(',')))
        logger.info(f'epoch {epoch}: {n} requires grad: {t.requires_grad}')


def main(args):
    local_config = json.load(open(args.local_config_path))
    local_config['loss'] = args.loss
    local_config['data_dir'] = args.data_dir
    local_config['train_batch_size'] = args.train_batch_size
    local_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    local_config['lr_scheduler'] = args.lr_scheduler
    local_config['model_name'] = args.model_name
    local_config['pool_type'] = args.pool_type
    local_config['seed'] = args.seed
    local_config['do_train'] = args.do_train
    local_config['do_validation'] = args.do_validation
    local_config['do_eval'] = args.do_eval
    local_config['use_cuda'] = args.use_cuda.lower() == 'true'
    local_config['num_train_epochs'] = args.num_train_epochs
    local_config['eval_batch_size'] = args.eval_batch_size
    local_config['max_seq_len'] = args.max_seq_len
    local_config['syns'] = ["Target", "Synonym"]
    local_config['target_embeddings'] = args.target_embeddings
    local_config['symmetric'] = args.symmetric.lower() == 'true'
    local_config['mask_syns'] = args.mask_syns
    local_config['train_scd'] = args.train_scd
    local_config['ckpt_path'] = args.ckpt_path
    local_config['head_batchnorm'] = args.head_batchnorm
    local_config['head_hidden_size'] = args.head_hidden_size
    local_config['linear_head'] = args.linear_head.lower() == 'true'
    local_config['emb_size_for_cosine'] = args.emb_size_for_cosine
    local_config['add_fc_layer'] = args.add_fc_layer

    if local_config['do_train'] and os.path.exists(args.output_dir):
        from glob import glob
        model_weights = glob(os.path.join(args.output_dir, '*.bin'))
        if model_weights:
            print(f'{model_weights}: already computed: skipping ...')
            return
        else:
            print(f'already existing {args.output_dir}. but without model weights ...')
            return

    device = torch.device("cuda" if local_config['use_cuda'] else "cpu")
    n_gpu = torch.cuda.device_count()

    if local_config['gradient_accumulation_steps'] < 1:
        raise ValueError(
            "gradient_accumulation_steps parameter should be >= 1"
        )

    local_config['train_batch_size'] = \
        local_config['train_batch_size'] // local_config['gradient_accumulation_steps']

    if local_config['do_train']:
        random.seed(local_config['seed'])
        np.random.seed(local_config['seed'])
        torch.manual_seed(local_config['seed'])

    if n_gpu > 0:
        torch.cuda.manual_seed_all(local_config['seed'])

    if not local_config['do_train'] and not local_config['do_eval']:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True."
        )

    if local_config['do_train'] and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'nen-nen-weights'))
    elif local_config['do_train'] or local_config['do_validation']:
        raise ValueError(args.output_dir, 'output_dir already exists')

    suffix = datetime.now().isoformat().replace('-', '_').replace(
        ':', '_').split('.')[0].replace('T', '-')
    if local_config['do_train']:
        train_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'train'
            )
        )
        dev_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'dev'
            )
        )

        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"train_{suffix}.log"), 'w')
        )
        eval_logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"scores_{suffix}.log"), 'w')
        )
    else:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.ckpt_path, f"eval_{suffix}.log"), 'w')
        )

    logger.info(args)
    logger.info(json.dumps(vars(args), indent=4))
    if args.do_train:
        json.dump(local_config, open(os.path.join(args.output_dir, 'local_config.json'), 'w'))
        json.dump(vars(args), open(os.path.join(args.output_dir, 'args.json'), 'w'))
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

#    with open(os.path.join(args.output_dir, 'local_config.json'), 'w') as outp:
#        json.dump(local_config, outp, indent=4)
#    with open(os.path.join(args.output_dir, 'args.json'), 'w') as outp:
#        json.dump(vars(args), outp, indent=4)


    syns = sorted(local_config['syns'])
    id2classifier = {i: classifier for i, classifier in enumerate(syns)}

    model_name = local_config['model_name']
    data_processor = DataProcessor()

    train_dir = os.path.join(local_config['data_dir'], 'train/')
    dev_dir = os.path.join(local_config['data_dir'], 'dev')
    
    if local_config['do_train']:

        config = configs[local_config['model_name']]
        config = config.from_pretrained(
            local_config['model_name'],
            hidden_dropout_prob=args.dropout
        )
        if args.ckpt_path != '':
            model_path = args.ckpt_path
        else:
            model_path = local_config['model_name']
        model = models[model_name].from_pretrained(
            model_path, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            local_config=local_config,
            data_processor=data_processor,
            config=config
        )

        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)
                ],
                'weight_decay': float(args.weight_decay)
            },
            {
                'params': [
                    param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(args.learning_rate),
            eps=1e-6,
            betas=(0.9, 0.98),
            correct_bias=True
        )

        train_features = model.convert_dataset_to_features(
            train_dir, logger
        )

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(
                train_features, key=lambda f: np.sum(f.input_mask)
            )
        else:
            random.shuffle(train_features)
#        import pdb; pdb.set_trace()
        train_dataloader = \
            get_dataloader_and_tensors(train_features, local_config['train_batch_size'])
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_batches) // local_config['gradient_accumulation_steps'] * \
                local_config['num_train_epochs']

        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        if local_config['lr_scheduler'] == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps
            )
        elif local_config['lr_scheduler'] == 'constant_warmup':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps
            )
        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", local_config['train_batch_size'])
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if local_config['do_validation']:
            dev_features = model.convert_dataset_to_features(
                dev_dir, logger
            )
            logger.info("***** Dev *****")
            logger.info("  Num examples = %d", len(dev_features))
            logger.info("  Batch size = %d", local_config['eval_batch_size'])
            dev_dataloader = \
                get_dataloader_and_tensors(dev_features, local_config['eval_batch_size'])
            test_dir = os.path.join(local_config['data_dir'], 'test/')
            if os.path.exists(test_dir):
                test_features = model.convert_dataset_to_features(
                    test_dir, test_logger
                )
                logger.info("***** Test *****")
                logger.info("  Num examples = %d", len(test_features))
                logger.info("  Batch size = %d", local_config['eval_batch_size'])

                test_dataloader = \
                    get_dataloader_and_tensors(test_features, local_config['eval_batch_size'])

        best_result = defaultdict(float)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        start_time = time.time()
        global_step = 0

        model.to(device)
        lr = float(args.learning_rate)
        for epoch in range(1, 1 + local_config['num_train_epochs']):
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            cur_train_loss = defaultdict(float)

            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, scheduler.get_lr()[0]))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)

            train_bar = tqdm(
                train_batches, total=len(train_batches),
                desc='training ... '
            )
            for step, batch in enumerate(
                train_bar
            ):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, \
                syn_labels, positions = batch
                train_loss, _ = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=input_mask,
                    input_labels={'syn_labels': syn_labels, 'positions': positions}
                )
                loss = train_loss['total'].mean().item()
                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()

                train_bar.set_description(f'training... [epoch == {epoch} / {local_config["num_train_epochs"]}, loss == {loss}]')

                loss_to_optimize = train_loss['total']

                if local_config['gradient_accumulation_steps'] > 1:
                    loss_to_optimize = \
                        loss_to_optimize / local_config['gradient_accumulation_steps']

                loss_to_optimize.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % local_config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if local_config['do_validation'] and (step + 1) % eval_step == 0:
                    logger.info(
                        'Ep: {}, Stp: {}/{}, usd_t={:.2f}s, loss={:.6f}'.format(
                            epoch, step + 1, len(train_batches),
                            time.time() - start_time, tr_loss / nb_tr_steps
                        )
                    )

                    cur_train_mean_loss = {}
                    for key, value in cur_train_loss.items():
                        cur_train_mean_loss[f'train_{key}_loss'] = \
                            value / nb_tr_steps

                    dev_predictions = os.path.join(args.output_dir, 'dev_predictions')

                    metrics = predict(
                        model, dev_dataloader, dev_predictions,
                        dev_features, args,
                        cur_train_mean_loss=cur_train_mean_loss,
                        logger=eval_logger
                    )

                    metrics['global_step'] = global_step
                    metrics['epoch'] = epoch
                    metrics['learning_rate'] = scheduler.get_lr()[0]
                    metrics['batch_size'] = \
                        local_config['train_batch_size'] * local_config['gradient_accumulation_steps']

                    for key, value in metrics.items():
                        dev_writer.add_scalar(key, value, global_step)
                    scores_to_logger = tuple([round(metrics[save_by_score] * 100.0, 2) for save_by_score in args.save_by_score.split('+')])
                    logger.info(f"dev %s (lr=%s, epoch=%d): %s" %
                        (
                            args.save_by_score,
                            str(scheduler.get_lr()[0]), epoch,
                            scores_to_logger
                        )
                    )

                    predict_parts = [part  for part in metrics if part.endswith('.score') and metrics[part] > args.start_save_threshold and metrics[part] > best_result[part]]
                    if len(predict_parts) > 0:
                        best_dev_predictions = os.path.join(args.output_dir, 'best_dev_predictions')
                        dev_predictions = os.path.join(args.output_dir, 'dev_predictions')
                        os.makedirs(best_dev_predictions, exist_ok=True)
                        for part in predict_parts:
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f -> %.2f" %
                                (
                                    part,
                                    str(scheduler.get_lr()[0]), epoch,
                                    best_result[part] * 100.0, metrics[part] * 100.0
                                )
                            )
                            best_result[part] = metrics[part]
                            if [save_weight for save_weight in args.save_by_score.split('+') if save_weight == part]:
                                os.makedirs(os.path.join(args.output_dir, part), exist_ok=True)
                                output_model_file = os.path.join(
                                    args.output_dir, part,
                                    WEIGHTS_NAME
                                )
                                save_model(args, model, output_model_file, metrics)
                            if 'nen-nen' not in part:
                                os.system(f'cp {dev_predictions}/{".".join(part.split(".")[1:-1])}* {best_dev_predictions}/')
                            else:
                                output_model_file = os.path.join(
                                    args.output_dir, 'nen-nen-weights',
                                    WEIGHTS_NAME
                                )
                                save_model(args, model, output_model_file, metrics)

                        # dev_predictions = os.path.join(args.output_dir, 'dev_predictions')
                        # predict(
                        #     model, dev_dataloader, dev_predictions,
                        #     dev_features, args, only_parts='+'.join(predict_parts)
                        # )
                        # best_dev_predictions = os.path.join(args.output_dir, 'best_dev_predictions')
                        # os.makedirs(best_dev_predictions, exist_ok=True)
                        # os.system(f'mv {dev_predictions}/* {best_dev_predictions}/')
                        if 'scd' not in '+'.join(predict_parts) and os.path.exists(test_dir):
                            test_predictions = os.path.join(args.output_dir, 'test_predictions')
                            test_metrics = predict(
                                model, test_dataloader, test_predictions,
                                test_features, args, only_parts='+'.join(['test' + part[3:] for part in predict_parts if 'nen-nen' not in part])
                            )
                            best_test_predictions = os.path.join(args.output_dir, 'best_test_predictions')
                            os.makedirs(best_test_predictions, exist_ok=True)
                            os.system(f'mv {test_predictions}/* {best_test_predictions}/')

                            for key, value in test_metrics.items():
                                if key.endswith('score'):
                                    dev_writer.add_scalar(key, value, global_step)


            if args.log_train_metrics:
                metrics = predict(
                    model, train_dataloader, os.path.join(args.output_dir, 'train_predictions'),
                    train_features, args,
                    logger=logger
                )
                metrics['global_step'] = global_step
                metrics['epoch'] = epoch
                metrics['learning_rate'] = scheduler.get_lr()[0]
                metrics['batch_size'] = \
                    local_config['train_batch_size'] * local_config['gradient_accumulation_steps']

                for key, value in metrics.items():
                    train_writer.add_scalar(key, value, global_step)

    if local_config['do_eval']:
        
        dm_model = DeepMistakeWiC(args.ckpt_path, device)
        
        assert args.ckpt_path != '', 'in do_eval mode ckpt_path should be specified'
        test_dir = args.eval_input_dir
        eval_output_dir = args.eval_output_dir
        output_dir = args.output_dir
        predictions = dm_model.predict_dataset(test_dir, output_dir, eval_output_dir)

        # config = configs[model_name].from_pretrained(model_name)
        # model = models[model_name].from_pretrained(args.ckpt_path, local_config=local_config, data_processor=data_processor, config=config)
        # model.to(device)
        # test_features = model.convert_dataset_to_features(
        #     test_dir, test_logger
        # )
        # logger.info("***** Test *****")
        # logger.info("  Num examples = %d", len(test_features))
        # logger.info("  Batch size = %d", local_config['eval_batch_size'])

        # test_dataloader = \
        #     get_dataloader_and_tensors(test_features, local_config['eval_batch_size'])

        # metrics = predict(
        #     model, test_dataloader,
        #     os.path.join(args.output_dir, args.eval_output_dir),
        #     test_features, args,
        #     compute_metrics=True
        # )
        # print(metrics)
        # with open(os.path.join(args.output_dir, args.eval_output_dir,'metrics.txt'), 'w') as outp:
        #     print(metrics, file=outp)


def save_model(args, model, output_model_file, metrics):
    start = time.time()
    model_to_save = \
        model.module if hasattr(model, 'module') else model

    output_config_file = os.path.join(
        args.output_dir, CONFIG_NAME
    )
    torch.save(
        model_to_save.state_dict(), output_model_file
    )
    model_to_save.config.to_json_file(
        output_config_file
    )
    with open(output_model_file+'.txt', 'w') as outp:
        print(metrics, file=outp)
    print(f'model saved in {time.time() - start} seconds to {output_model_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='data/', type=str,
                        help="The directory where train and dev directories are located.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_per_epoch", default=4, type=int,
                        help="How many times to do validation on dev set per epoch")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup.\n"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="train batch size")
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int,
                        help="gradinent accumulation steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="maximal gradient norm")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate")
    parser.add_argument("--local_config_path", type=str, default='local_config.json',
                        help="local config path")
    parser.add_argument("--start_save_threshold", default=0.0, type=float,
                        help="accuracy threshold to start save models")
    parser.add_argument("--log_train_metrics", action="store_true",
                        help="compute metrics for train set too")
    parser.add_argument("--loss", type=str, default='crossentropy_loss',
                        choices=['crossentropy_loss', 'mse_loss', 'cosine_similarity', 'mseplus_loss'])
    parser.add_argument("--lr_scheduler", type=str, default='linear_warmup',
                        choices=['constant_warmup', 'linear_warmup'])
    parser.add_argument("--model_name", type=str, default='xlm-roberta-large',
                        choices=['xlm-roberta-large', 'xlm-roberta-base'])

    parser.add_argument("--pool_type", type=str, default='first')

    parser.add_argument("--save_by_score", type=str, default='accuracy.dev.en-en.score')
    parser.add_argument("--ckpt_path", type=str, default='', help='Path to directory containig pytorch.bin checkpoint')
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--target_embeddings", type=str, default='concat')
    parser.add_argument("--head_batchnorm", type=int, default=0)
    parser.add_argument("--head_hidden_size", type=int, default=-1)
    parser.add_argument("--linear_head", type=str, default='false')
    parser.add_argument("--add_fc_layer", type=str, default='True')
    parser.add_argument("--emb_size_for_cosine", type=int, default=1024)

    parser.add_argument("--do_train", action='store_true', help='Whether to run training')
    parser.add_argument("--do_validation", action='store_true',
                        help='Whether to validate model during training process')
    parser.add_argument("--do_eval", action='store_true', help='Whether to run evaluation')
    parser.add_argument("--use_cuda", default='true', type=str, help='Whether to use GPU')
    parser.add_argument("--symmetric", default='true', type=str, help='Whether to augment data by symmetry')
    parser.add_argument("--mask_syns", action='store_true',
                        help='Whether to replace target words in context by mask tokens')
    parser.add_argument("--train_scd", action='store_true', help='Whether to train semantic change detection model')
    parser.add_argument("--eval_input_dir", default='data/wic/test', help='Directory containing .data files to predict')
    parser.add_argument("--eval_output_dir", default='best_eval_test_predictions', help='Directory name where predictions will be saved')

    parser.add_argument("--trainable_params", default='', help='Comma-separated list of substrings of trainable parameter names; optionally, after a colon the first epoch this hyperparameters becomes trainable')

    parsed_args = parser.parse_args()
    if parsed_args.do_eval:
        try:
            new_args = json.load(open(os.path.join(parsed_args.output_dir, 'args.json')))
        except FileNotFoundError:
            # os print current directory
            print(os.getcwd())
            new_args = json.load(open(os.path.join(parsed_args.output_dir, '/args.json')))
        for key, value in new_args.items():
            if key.startswith('do') or key in ['ckpt_path', 'eval_input_dir', 'eval_output_dir', 'output_dir', 'eval_batch_size','use_cuda']:
                continue
            setattr(parsed_args, key, value)
    main(parsed_args)
