from .xlmr import XLMRModel
from transformers import XLMRobertaConfig
from .utils import Example, get_dataloader_and_tensors, DataProcessor
import json
import os
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.special import softmax
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn import CosineSimilarity
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("__scores__")
test_logger = logging.getLogger("__test__")


class DeepMistakeWiC:

    def __init__(self, ckpt_dir, device):
        """
        Loads a DeepMistake model from the specified checkpoint directory.
        """
        self.device = device
        data_processor = DataProcessor()

        with open(str(ckpt_dir) + "/local_config.json", "r") as f:
            self.local_config = json.load(f)

        self.config = XLMRobertaConfig.from_pretrained(self.local_config["model_name"])
        self.model = XLMRModel.from_pretrained(
            ckpt_dir,
            data_processor=data_processor,
            config=self.config,
            local_config=self.local_config,
        ).to(device)

    def predict_examples(self, examples: list[Example], log, batch_size=16):
        """
        Runs inference for the provided list of examples.
        """
        # Runs inference, similarly to run_model.py, lines 624-634
        test_features = self.model.convert_examples_to_features(examples, log)
        eval_dataloader = get_dataloader_and_tensors(test_features, batch_size)
        syns_preds, syns_scores_res, metrics = self.predict(
            model=self.model,
            eval_dataloader=eval_dataloader,
            eval_fearures=test_features,
            compute_metrics=False,
            dump_feature=False,
            output_dir="dm_testwug",
        )
        if self.local_config["symmetric"]:
            # two cosecutive examples should be averaged
            syns_scores_res = np.array(
                [
                    (syns_scores_res[i] + syns_scores_res[i + 1]) / 2
                    for i in range(0, len(syns_scores_res), 2)
                ]
            )
            syns_preds = np.zeros(syns_scores_res.shape, dtype=int)
            syns_preds[syns_scores_res > 0.5] = 1
            
        return syns_scores_res, syns_preds

    def predict_dataset(self, test_dir, output_dir, eval_output_dir):
        """
        Runs inference for the provided dataset.
        """
        test_features = self.model.convert_dataset_to_features(test_dir, test_logger)
        eval_dataloader = get_dataloader_and_tensors(
            test_features, self.local_config["eval_batch_size"]
        )

        metrics, syns_preds, syns_scores_res = self.predict(
            self.model,
            eval_dataloader,
            os.path.join(output_dir, eval_output_dir),
            test_features,
            compute_metrics=True,
        )

        print(metrics)
        with open(
            os.path.join(output_dir, eval_output_dir, "metrics.txt"), "w"
        ) as outp:
            print(metrics, file=outp)

    def predict(
        self,
        model,
        eval_dataloader,
        output_dir,
        eval_fearures,
        cur_train_mean_loss=None,
        compute_metrics=True,
        only_parts="",
        dump_feature=True,
    ):

        if os.path.exists(output_dir):
            os.system(f"rm -r {output_dir}/*")
        else:
            os.makedirs(output_dir, exist_ok=True)

        only_parts = [part for part in only_parts.split("+") if part]
        model.eval()
        syns = sorted(model.local_config["syns"])

        eval_dataloader = tqdm(
            eval_dataloader, total=len(eval_dataloader), desc="validation ... "
        )
        syns_preds, syns_scores_res, metrics = self.run_inference(
            eval_dataloader,
            model,
            self.device,
            compute_metrics,
            dump_feature,
            output_dir,
        )

        predictions = defaultdict(lambda: defaultdict(list))
        golds = defaultdict(lambda: defaultdict(list))
        scores = defaultdict(lambda: defaultdict(list))
        gold_scores = defaultdict(lambda: defaultdict(list))
        lemmas = defaultdict(lambda: defaultdict(list))

        syn_ids_to_label = {0: "F", 1: "T"}
        for ex_id, (ex_feature, ex_syn_preds, ex_scores) in enumerate(
            zip(eval_fearures, syns_preds, syns_scores_res)
        ):
            example = ex_feature.example
            docId = example.docId
            posInDoc = int(docId.split(".")[-1])
            docId = ".".join(docId.split(".")[:-1])
            syn_pred = syn_ids_to_label[ex_syn_preds.item()]
            predictions[docId][posInDoc].append(syn_pred)
            golds[docId][posInDoc].append(example.label)
            scores[docId][posInDoc].append(ex_scores)
            gold_scores[docId][posInDoc].append(example.score)
            lemmas[docId][posInDoc].append((example.lemma, example.grp))

        print(f"saving predictions for: {only_parts}")
        for docId, doc_preds in predictions.items():
            doc_scores = scores[docId]
            if len(only_parts) > 0 and all(
                [f'{docId.split(".")[1]}.score' not in part for part in only_parts]
            ):
                continue
            print(f"saving predictions for part: {docId}")
            prediction = [
                {"id": f"{docId}.{pos}", "tag": "F" if "F" in doc_preds[pos] else "T"}
                for pos in sorted(doc_preds)
            ]
            prediction_file = os.path.join(output_dir, docId)
            json.dump(prediction, open(prediction_file, "w"))
            prediction = [
                {"id": f"{docId}.{pos}", "score": [str(x) for x in doc_scores[pos]]}
                for pos in sorted(doc_preds)
            ]
            prediction_file = os.path.join(output_dir, f"{docId}.scores")
            json.dump(prediction, open(prediction_file, "w"))

        if compute_metrics:
            mean_non_english = []
            for docId, doc_preds in predictions.items():
                if "scd" in docId:
                    doc_golds = gold_scores[docId]
                    doc_lemmas = lemmas[docId]
                    doc_scores = scores[docId]

                    keys = sorted(list(doc_golds.keys()))
                    # print(doc_lemmas)
                    unique_lemmas = sorted(
                        set(
                            [
                                doc_lemmas[key][0][0]
                                for key in keys
                                if doc_lemmas[key][0][1] == "COMPARE"
                            ]
                        )
                    )
                    y_true, y_pred = [], []
                    y_sent_true, y_sent_pred = [], []
                    for unique_lemma in unique_lemmas:
                        unique_lemma_keys = [
                            key
                            for key in keys
                            if doc_lemmas[key][0][0] == unique_lemma
                            and doc_lemmas[key][0][1] == "COMPARE"
                        ]
                        unique_word_scores_pred = [
                            np.array(doc_scores[key]).mean()
                            for key in unique_lemma_keys
                        ]
                        unique_word_scores_true = [
                            doc_golds[key][0] for key in unique_lemma_keys
                        ]
                        y_true.append(np.array(unique_word_scores_true).mean())
                        y_pred.append(np.array(unique_word_scores_pred).mean())
                        y_sent_true.extend(unique_word_scores_true)
                        y_sent_pred.extend(unique_word_scores_pred)
                    metrics[f"spearman.{docId}.wordwise.score"], _ = spearmanr(
                        y_true, y_pred
                    )
                    metrics[f"spearman.{docId}.score"], _ = spearmanr(
                        y_sent_true, y_sent_pred
                    )
                    doc_golds = golds[docId]
                    keys = list(doc_golds.keys())
                    doc_golds = [doc_golds[key][0] for key in keys]
                    doc_preds = ["F" if "F" in doc_preds[key] else "T" for key in keys]
                    metrics[f"{docId}.accuracy"] = accuracy_score(doc_golds, doc_preds)
                else:
                    doc_golds = golds[docId]
                    keys = list(doc_golds.keys())
                    doc_golds = [doc_golds[key][0] for key in keys]
                    doc_preds = ["F" if "F" in doc_preds[key] else "T" for key in keys]
                    metrics[f"accuracy.{docId}.score"] = accuracy_score(
                        doc_golds, doc_preds
                    )
                    if "en-en" not in docId:
                        mean_non_english.append(metrics[f"accuracy.{docId}.score"])
            if mean_non_english:
                metrics[f'accuracy.{docId.split(".")[0]}.nen-nen.score'] = sum(
                    mean_non_english
                ) / len(mean_non_english)

            if cur_train_mean_loss is not None:
                metrics.update(cur_train_mean_loss)
        else:
            metrics = {}

        model.train()

        return syns_preds, syns_scores_res, metrics

    def run_inference(
        self, eval_dataloader, model, device, compute_metrics, dump_feature, output_dir
    ):
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
                    input_labels={"syn_labels": b_syn_labels, "positions": b_positions},
                    return_features=True,
                )

            if dump_feature:
                syn_features = syn_features.detach().cpu().numpy()
                np.save(
                    os.path.join(output_dir, f"features_batch_{batch_id}.npy"),
                    syn_features,
                )
                np.save(
                    os.path.join(output_dir, f"labels_batch_{batch_id}.npy"),
                    b_syn_labels.detach().cpu().numpy(),
                )

            if compute_metrics:
                for key, value in loss.items():
                    metrics[f"eval_{key}_loss"] = (
                        metrics[f"eval_{key}_loss"] * nb_eval_steps
                        + value.mean().item()
                    ) / (nb_eval_steps + 1)

            nb_eval_steps += 1
            if model.local_config["loss"] != "cosine_similarity":
                syns_preds.append(syn_logits.detach().cpu().numpy())
            else:
                syns_preds.append(
                    CosineSimilarity()(syn_logits[0], syn_logits[1])
                    .detach()
                    .cpu()
                    .numpy()
                )
        syns_scores = np.concatenate(syns_preds, axis=0)  # n_examples x 2 or n_examples
        if model.local_config["loss"] == "cosine_similarity":
            syns_scores_res = syns_scores
        elif syns_scores.ndim > 1 and syns_scores.shape[-1] > 1:
            syns_scores_res = softmax(syns_scores, axis=-1)[:, -1]
        else:
            syns_scores_res = syns_scores[:, 0]
        # Create predictions from scores
        if (
            syns_scores.shape[-1] != 1
            and model.local_config["loss"] != "cosine_similarity"
        ):
            syns_preds = np.argmax(syns_scores, axis=1)  # n_examples
        elif model.local_config["loss"] == "cosine_similarity":
            syns_preds = np.zeros(syns_scores.shape, dtype=int)
            syns_preds[syns_scores >= 0.5] = 1
        else:
            syns_preds = np.zeros(syns_scores.shape, dtype=int)
            if model.local_config["train_scd"]:
                syns_preds[syns_scores >= 3.0] = 1
            else:
                syns_preds[syns_scores > 0.5] = 1
        return syns_preds, syns_scores_res, metrics