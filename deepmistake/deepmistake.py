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
        ).to(self.device)

    def predict_examples(self, examples: list[Example], log, batch_size=16):
        """
        Runs inference for the provided list of examples.
        """
        # Runs inference, similarly to run_model.py, lines 624-634
        test_features = self.model.convert_examples_to_features(examples, log)
        eval_dataloader = get_dataloader_and_tensors(test_features, batch_size)
        syns_preds, syns_scores_res = self.predict(
            model=self.model,
            eval_dataloader=eval_dataloader,
        )
        return syns_scores_res, syns_preds

    def predict_dataset(self, test_dir, output_dir, eval_output_dir):
        """
        Runs inference for the provided dataset.
        """
        test_features = self.model.convert_dataset_to_features(test_dir, test_logger)
        eval_dataloader = get_dataloader_and_tensors(
            test_features, self.local_config["eval_batch_size"]
        )

        syns_preds, syns_scores_res = self.predict(
            self.model,
            eval_dataloader,
        )

        return syns_scores_res, syns_preds

    def predict(
        self,
        model,
        eval_dataloader,
        only_parts="",
    ):

        only_parts = [part for part in only_parts.split("+") if part]
        model.eval()
        syns = sorted(model.local_config["syns"])

        eval_dataloader = tqdm(
            eval_dataloader, total=len(eval_dataloader), desc="validation ... "
        )
        syns_preds, syns_scores_res = self.run_inference(
            eval_dataloader,
            model,
            self.device,
        )

        return syns_preds, syns_scores_res

    def run_inference(self, eval_dataloader, model, device):
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
        return syns_preds, syns_scores_res
