from transformers import RobertaModel, XLMRobertaConfig
from transformers import BertPreTrainedModel
from transformers import XLMRobertaTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_classes, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationCosineHead(nn.Module):
    def __init__(self, config, num_classes, input_size, local_config):
        super().__init__()
        self.emb_size = local_config['emb_size_for_cosine']
        self.input_size=input_size
        print(f"INPUT_SIZE======{self.input_size}")
        self.emb_fs = nn.Linear(int(input_size // 2), self.emb_size)
        self.emb_sc = nn.Linear(int(input_size // 2), self.emb_size)
        self.activation = nn.Tanh()
        self.local_config = local_config
        self.cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def calcSim(self, emb1, emb2):
        return self.cos_fn(emb1, emb2)
        
    def forward(self, features, **kwargs):
        emb1 = features[:, :int(self.input_size // 2)]
        emb2 = features[:, int(self.input_size // 2):self.input_size]
        if self.local_config['add_fc_layer'] == "True":
            emb1 = self.activation(self.emb_fs(emb1))
            emb2 = self.activation(self.emb_sc(emb2))
        return (emb1, emb2)


class WiCFeature2:
    def __init__(self, input_ids, input_mask, token_type_ids, syn_label, positions, example):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.syn_label = syn_label
        self.positions = positions
        self.example = example


class XLMRModel(BertPreTrainedModel):
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
        self.roberta = RobertaModel(config)
        self.local_config = local_config
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(local_config['model_name'])
        input_size = config.hidden_size
        if local_config['target_embeddings'] == 'concat':
            input_size *= 2
        if self.local_config['loss'] == 'mse_loss':
            self.syn_mse_clf = RobertaClassificationHead(config, 1, input_size)
        elif self.local_config['loss'] == 'crossentropy_loss':
            self.syn_clf = RobertaClassificationHead(config, 2, input_size)
        elif self.local_config['loss'] == 'cosine_similarity':
            self.syn_clf = RobertaClassificationCosineHead(config, 2, input_size, local_config)
        self.data_processor = data_processor
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
    ):
        loss = defaultdict(float)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequences_output = outputs[0]  # bs x seq x hidden

        syn_labels = input_labels['syn_labels']  # bs
        positions = input_labels['positions'] # bs x 4

        syn_features = self.extract_features(sequences_output, positions) # bs x hidden
        clf = self.syn_mse_clf if self.local_config['loss'] == 'mse_loss' else self.syn_clf
        syn_logits = clf(syn_features)  # bs x 2 or bs

        if input_labels is not None:
            if self.local_config['loss'] != 'cosine_similarity':
                y_size = syn_logits.size(-1)
            else:
                y_size = -1
            if y_size == 1:
                loss['total'] = MSELoss()(syn_logits, syn_labels.unsqueeze(-1).float())
            elif self.local_config['loss'] == 'crossentropy_loss':
                loss['total'] = CrossEntropyLoss()(syn_logits, syn_labels)
            else:
                loss['total'] = CosineEmbeddingLoss()(syn_logits[0], syn_logits[1], syn_labels * 2 - 1)

        return (loss, syn_logits)


    def extract_features(self, hidden_states, positions):
        pool_type = self.local_config['pool_type']
        merge_type = self.local_config['target_embeddings']
        bs, seq, hs = hidden_states.size()
        features = []
        for ex_id in range(bs):
            start1, end1, start2, end2 = positions[ex_id, 0].item(), positions[ex_id, 1].item(), positions[ex_id, 2].item(), positions[ex_id, 3].item()
            if pool_type == 'mean':
                emb1 = hidden_states[ex_id, start1:end1].mean(dim=0) # hidden
                emb2 = hidden_states[ex_id, start2:end2].mean(dim=0) # hidden
            elif pool_type == 'max':
                emb1, _ = hidden_states[ex_id, start1:end1].max(dim=0) # hidden
                emb2, _ = hidden_states[ex_id, start2:end2].max(dim=0) # hidden
            elif pool_type == 'first':
                emb1 = hidden_states[ex_id, start1]
                emb2 = hidden_states[ex_id, start2]
            else:
                raise ValueError(f'wrong pool_type: {pool_type}')
            if merge_type == 'featwise_mul':
                merged_feature = emb1 * emb2 # hidden
            elif merge_type == 'concat':
                merged_feature = torch.cat((emb1, emb2)) # 2 * hidden
            features.append(merged_feature.unsqueeze(0))
        output = torch.cat(features, dim=0) # bs x hidden
        return output


    def convert_dataset_to_features(
            self, source_dir, logger
    ):
        features = []
        max_seq_len = self.local_config['max_seq_len']

        examples = self.data_processor.get_examples(source_dir)
        syns = self.local_config['syns']
        syn_label_to_id = {'T': 1, 'F': 0}
        syns_lab_to_pos = {lab: i for i, lab in enumerate(syns)}
        num_too_long_exs = 0

        for (ex_index, ex) in enumerate(examples):
            pos, label = ex.pos, ex.label
            for i, (st1, end1, sent1, st2, end2, sent2) in enumerate([(ex.start_1, ex.end_1, ex.text_1, ex.start_2, ex.end_2, ex.text_2), (ex.start_2, ex.end_2, ex.text_2, ex.start_1, ex.end_1, ex.text_1)]):
                if not self.local_config['symmetric'] and i == 0:
                    continue
                st1, end1, st2, end2 = int(st1), int(end1), int(st2), int(end2)
                tokens = [self.tokenizer.cls_token]

                positions = [0] * (2 * len(syns))
                left1, target1, right1 = sent1[:st1], sent1[st1:end1], sent1[end1:]
                left2, target2, right2 = sent2[:st2], sent2[st2:end2], sent2[end2:]

                if left1:
                    tokens += self.tokenizer.tokenize(left1)
                positions[syns_lab_to_pos['Target'] * 2] = len(tokens)
                target_subtokens = self.tokenizer.tokenize(target1)
                if self.local_config['mask_syns']:
                    tokens += [self.tokenizer.mask_token] * len(target_subtokens)
                else:
                    tokens += target_subtokens
                positions[syns_lab_to_pos['Target'] * 2 + 1] = len(tokens)

                if right1:
                    tokens += self.tokenizer.tokenize(right1) + [self.tokenizer.sep_token]
                if left2:
                    tokens += self.tokenizer.tokenize(left2)

                positions[syns_lab_to_pos['Synonym'] * 2] = len(tokens)
                target_subtokens = self.tokenizer.tokenize(target2)
                if self.local_config['mask_syns']:
                    tokens += [self.tokenizer.mask_token] * len(target_subtokens)
                else:
                    tokens += target_subtokens
                positions[syns_lab_to_pos['Synonym'] * 2 + 1] = len(tokens)
                if right2:
                    tokens += self.tokenizer.tokenize(right2) + [self.tokenizer.sep_token]

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                    num_too_long_exs += 1
                    if max(positions) > max_seq_len - 1:
                        continue

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
                    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info("label: %s" % label)
                    for clf_id, clf in enumerate(syns):
                        start, end = positions[clf_id * 2: clf_id * 2 + 2]
                        if start and end:
                            logger.info(f"{clf}: {' '.join(tokens[start:end])}")
                if self.local_config['train_scd']:
                    assert self.local_config['loss'] == 'mse_loss', 'should be mse loss when training scd'
                    features.append(
                        WiCFeature2(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            syn_label=ex.score,
                            positions=positions,
                            example=ex
                            )
                        )
                else:
                    features.append(
                        WiCFeature2(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            syn_label=syn_label_to_id[label],
                            positions=positions,
                            example=ex
                        )
                    )
        logger.info("Not fitted examples percentage: %s" % str(num_too_long_exs / len(features) * 100.0))
        return features
