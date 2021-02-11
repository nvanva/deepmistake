from transformers import RobertaModel, XLMRobertaConfig
from transformers import BertPreTrainedModel
from transformers import XLMRobertaTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
from collections import defaultdict


XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                 "-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                  "-large-finetuned-conll03-german-pytorch_model.bin",
}


class WiCFeature:
    def __init__(self, input_ids, input_mask, token_type_ids, syn_labels, pos_label, example, min_pos):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.syn_labels = syn_labels
        self.pos_label = pos_label
        self.example = example
        self.min_pos = min_pos


class XLMRQAModel(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(
            self,
            config: XLMRobertaConfig,
            local_config: dict,
            data_processor
        ):
        super().__init__(config)
        syns = sorted(local_config['syns'])
        self.num_clfs = len(syns) + 1 if local_config['train_pos'] else len(syns)
        self.clfs_weights = torch.nn.parameter.Parameter(torch.ones(self.num_clfs, dtype=torch.float32), requires_grad=True)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.local_config = local_config
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(local_config['model_name'])
        self.clf2ncls = [2 for clf in syns]
        assert local_config['target_embeddings'] in ['concat', 'none']
        if local_config['target_embeddings'] == 'concat':
            self.syns = nn.Linear(config.hidden_size * 2, len(syns) * 2)
            self.pos_clf = nn.Linear(config.hidden_size * 2, self.local_config['pos_ncls'])
        else:
            self.syns = nn.Linear(config.hidden_size, len(syns) * 2)
            self.pos_clf = nn.Linear(config.hidden_size, self.local_config['pos_ncls'])
        print(self.clfs_weights)

        self.data_processor = data_processor

        self.TARGET_START = '•'
        self.TARGET_END = '⁄'

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_labels=None,
            weighting_mode='softmax'
    ):
        assert weighting_mode in ['softmax', 'rsqr+log', 'equal'], f'wrong weighting_mode: {weighting_mode}'
        loss = defaultdict(float)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        syns = sorted(self.local_config['syns'])
        sequences_output = self.dropout(outputs[0])  # bs x seq x hidden
        weights = self.clfs_weights

        labels = input_labels['labels']  # bs x 4
        pos_label = input_labels['pos_label']  # bs

        syn_labels = torch.split(labels, 2, dim=-1)  # 2 x bs x 2

        if self.local_config['target_embeddings'] == 'concat':
            pos = [i for i, syn_clf in enumerate(syns) if syn_clf == 'Target'][0]
            sequences_output = self.concat_target_embeddings(sequences_output, syn_labels[pos])

        syns_logits = self.syns(sequences_output)  # bs x seq x 4
        pos_logits = self.pos_clf(sequences_output[:, 0, :])  # bs x pos_ncls
        syn_clfs_logits = torch.split(syns_logits, self.clf2ncls, dim=-1) # 2 x bs x seq x 2

        if input_labels is not None:
            ignore_index = syns_logits.size(1)
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            pos_loss_fct = CrossEntropyLoss()
            labels.clamp_(0, ignore_index)

            for clf_id, (syn_clf_logits, syn_clf_labels) in enumerate(zip(syn_clfs_logits, syn_labels)):
                clf_loss = 0.0
                for clf_logit, clf_label in zip(torch.split(syn_clf_logits, 1, dim=-1), torch.split(syn_clf_labels, 1, dim=1)):
                    clf_logit = clf_logit.squeeze(-1)  # bs x seq
                    clf_label = clf_label.squeeze(-1)  # bs
                    clf_loss += 1.0 / syn_clf_labels.size(-1) * loss_fct(clf_logit, clf_label)
                loss[syns[clf_id]] = clf_loss

                if weighting_mode == 'softmax':
                    loss['total'] += torch.exp(weights[clf_id]) * clf_loss
                elif weighting_mode == 'rsqr+log':
                    loss['total'] += 1. / (weights[clf_id] ** 2) * clf_loss + torch.log(weights[clf_id])
                elif weighting_mode == 'equal':
                    loss['total'] += 1. / self.num_clfs * clf_loss

            pos_loss = pos_loss_fct(pos_logits, pos_label)
            if not self.local_config['train_pos']:
                assert weighting_mode == 'equal'
                pos_loss *= 0.0
            loss['pos'] = pos_loss
            if weighting_mode == 'softmax':
                loss['total'] += torch.exp(weights[-1]) * pos_loss
            elif weighting_mode == 'rsqr+log':
                loss['total'] += 1. / (weights[-1] ** 2) * pos_loss + torch.log(weights[-1])
            elif weighting_mode == 'equal':
                loss['total'] += 1. / self.num_clfs * pos_loss

            if weighting_mode == 'softmax':
                loss['total'] /= torch.sum(torch.exp(weights))

        outputs = (loss, syns_logits, pos_logits)

        return outputs


    def concat_target_embeddings(self, hidden_states, target_labels):
        pool_type = self.local_config['pool_type']
        bs, seq, hs = hidden_states.size()
        device = torch.device('cuda') if self.local_config['use_cuda'] else torch.device('cpu')
        output = torch.zeros(bs, seq, 2 * hs, dtype=torch.float32, device=device)
        output[:, :, :hs] = hidden_states
        for ex_id in range(bs):
            start, end = target_labels[ex_id, 0].item(), target_labels[ex_id, 1].item()
            quant_emb = 0.0
            if pool_type == 'mean':
                quant_emb = hidden_states[ex_id, start:end+1].mean(dim=0)
            elif pool_type == 'max':
                quant_emb, _ = hidden_states[ex_id, start:end+1].max(dim=0)
            else:
                raise ValueError(f'wrong pool_type: {pool_type}')
            output[:, :, hs:] = quant_emb
        return output


    def convert_dataset_to_features(
            self, source_dir, logger, add_question='Find the same sense of the marked word'
    ):
        syns = sorted(self.local_config['syns'])
        if add_question:
            question_tokens = self.tokenizer.tokenize(add_question)
            logger.info(f'question_tokens: {question_tokens}')

        features = []
        max_seq_len = self.local_config['max_seq_len']

        pos_lab_to_id = {lab: i for i, lab in enumerate(sorted(self.local_config['pos_classes']))}
        pos_id_to_lab = {i: lab for i, lab in enumerate(sorted(self.local_config['pos_classes']))}
        logger.info(f'pos_lab_to_id: {pos_lab_to_id}')

        examples = self.data_processor.get_examples(source_dir)
        syns_lab_to_pos = {lab: i for i, lab in enumerate(syns)}
        num_too_long_exs = 0

        for (ex_index, ex) in enumerate(examples):
            pos, label = ex.pos, ex.label
            for i, (st1, end1, sent1, st2, end2, sent2) in enumerate([(ex.start_1, ex.end_1, ex.text_1, ex.start_2, ex.end_2, ex.text_2), (ex.start_2, ex.end_2, ex.text_2, ex.start_1, ex.end_1, ex.text_1)]):
                if not self.local_config['symmetric'] and i == 0:
                    continue
                st1, end1, st2, end2 = int(st1), int(end1), int(st2), int(end2)
                tokens = [self.tokenizer.cls_token]
                if add_question:
                    tokens += question_tokens + [self.tokenizer.sep_token]
                min_pos = len(tokens)
                pos_label = pos_lab_to_id[ex.pos]
                syn_labels = [0] * (2 * len(syns))
                left1, target1, right1 = sent1[:st1], sent1[st1:end1], sent1[end1:]
                left2, target2, right2 = sent2[:st2], sent2[st2:end2], sent2[end2:]
                if left1:
                    tokens += self.tokenizer.tokenize(left1)
                tokens += [self.TARGET_START]
                syn_labels[syns_lab_to_pos['Target'] * 2] = len(tokens)
                tokens += self.tokenizer.tokenize(target1)
                syn_labels[syns_lab_to_pos['Target'] * 2 + 1] = len(tokens) - 1
                tokens += [self.TARGET_END]
                if right1:
                    tokens += self.tokenizer.tokenize(right1) + [self.tokenizer.sep_token]
                if left2:
                    tokens += self.tokenizer.tokenize(left2)
                if label != 'F':
                    syn_labels[syns_lab_to_pos['Synonym'] * 2] = len(tokens)
                tokens += self.tokenizer.tokenize(target2)
                if label != 'F':
                    syn_labels[syns_lab_to_pos['Synonym'] * 2 + 1] = len(tokens) - 1
                if right2:
                    tokens += self.tokenizer.tokenize(right2) + [self.tokenizer.sep_token]

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    num_too_long_exs += 1

                input_mask = [1] * len(input_ids)
                token_type_ids = [0] * max_seq_len
                padding = [self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]] * (max_seq_len - len(input_ids))
                input_ids += padding
                input_mask += [0] * len(padding)

                if ex_index % 10000 == 0:
                    if self.local_config['symmetric']:
                        logger.info("Writing example %d of %d" % (ex_index + i, len(examples) * 2))
                    else:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                if ex_index < 10:
                    logger.info("*** Example ***")
                    logger.info("subtokens: %s" % " ".join(
                        [x for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("pos label: %s" % pos_id_to_lab[pos_label])
                    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    for clf_id, clf in enumerate(syns):
                        start, end = syn_labels[clf_id * 2: clf_id * 2 + 2]
                        if start and end:
                            logger.info(f"{clf} label: {' '.join(tokens[start:end+1])}")

                features.append(
                    WiCFeature(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        token_type_ids=token_type_ids,
                        syn_labels=syn_labels,
                        pos_label=pos_label,
                        example=ex,
                        min_pos=min_pos
                    )
                )
        logger.info("Not fitted examples percentage: %s" % str(num_too_long_exs / (len(examples) * 2) * 100.0))
        return features
