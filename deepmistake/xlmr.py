from transformers import RobertaModel, XLMRobertaConfig
from transformers import BertPreTrainedModel
from transformers import XLMRobertaTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
import torch
from collections import defaultdict
from torch import Tensor
import torch.nn.functional as F
import re
from math import ceil 

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

    def __init__(self, config, num_classes, input_size, local_config):
        super().__init__()
        bn = local_config['head_batchnorm']
        self.linear_head = local_config['linear_head'] 
        hidden_size = local_config['head_hidden_size']
        hidden_size = config.hidden_size if hidden_size <=0 else hidden_size
        self.bn1 = torch.nn.BatchNorm1d(input_size) if bn%2==1 else None
        self.bn2 = torch.nn.BatchNorm1d(hidden_size) if bn//2==1 else None
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(input_size if self.linear_head else hidden_size, num_classes)
        print(f'RobertaClassificationHead: linear_head={self.linear_head}, hs={hidden_size}, input={input_size}, bn={bn}')

    def forward(self, features, **kwargs):
        x = features if self.bn1 is None else self.bn1(features)
        if not self.linear_head:
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = x if self.bn2 is None else self.bn2(x)
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

class MSEPlusLoss(MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', pos_score=3.5, neg_score=1.5, eps=0.5) -> None:
        super(MSEPlusLoss, self).__init__(size_average, reduce, reduction)
        self.pos_score, self.neg_score, self.eps = pos_score, neg_score, eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
#        import pdb; pdb.set_trace()
        input = input + 2.5
        target_score = -target * (self.pos_score-self.neg_score) + self.neg_score  # neg_score if target==0, pos_score if target==-1, otherwise doesn't matter
        diffplus = torch.max(F.relu(target_score-self.eps-input), F.relu(input-target_score-self.eps)) 
        diff = torch.where(target > 0.0, input-target, diffplus)
        print(' '.join(f'{a:.2f}({b})->{c:.2f}' for a,b,c in zip(input.flatten().tolist(), target.flatten().tolist(), diff.flatten().tolist())), flush=True)
        return nn.functional.mse_loss(diff, torch.zeros_like(target), reduction=self.reduction)


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

        if local_config['pool_type'] in {'mmm','mmf'}:
            input_size *= 3
        elif local_config['pool_type'] in {'mm','mf'}:
            input_size *= 2

        if local_config['target_embeddings'] == 'concat':
            input_size *= 2
        elif local_config['target_embeddings'].startswith('comb_c'):
            input_size *= 3
        elif local_config['target_embeddings'].startswith('comb_'):
            input_size *= 2
        elif local_config['target_embeddings'].startswith('dist_'):
            input_size = len(local_config['target_embeddings'].replace('dist_','').replace('n1','').replace('n',''))//2 

        print('Classification head input size:', input_size)
        if self.local_config['loss'] in {'mseplus_loss', 'mse_loss'}:
            self.syn_mse_clf = RobertaClassificationHead(config, 1, input_size, self.local_config)
        elif self.local_config['loss'] == 'crossentropy_loss':
            self.syn_clf = RobertaClassificationHead(config, 2, input_size, self.local_config)
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
            return_features=False
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
        clf = self.syn_mse_clf if self.local_config['loss'] in {'mseplus_loss','mse_loss'} else self.syn_clf
        syn_logits = clf(syn_features)  # bs x 2 or bs

        if input_labels is not None:
            if self.local_config['loss'] != 'cosine_similarity':
                y_size = syn_logits.size(-1)
            else:
                y_size = -1
            if y_size == 1:
                lossfn = MSELoss() if self.local_config['loss'] == 'mse_loss' else MSEPlusLoss()
                loss['total'] = lossfn(syn_logits, syn_labels.unsqueeze(-1).float())
            elif self.local_config['loss'] == 'crossentropy_loss':
                loss['total'] = CrossEntropyLoss()(syn_logits, syn_labels)
            else:
                loss['total'] = CosineEmbeddingLoss()(syn_logits[0], syn_logits[1], syn_labels * 2 - 1)

        return (loss, syn_logits, syn_features) if return_features else (loss, syn_logits)


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
            elif pool_type == 'mf':
                embs1 = hidden_states[ex_id, start1:end1] # hidden
                embs2 = hidden_states[ex_id, start2:end2] # hidden
                emb1, emb2 = (torch.cat([embs.mean(dim=0), embs[0]], dim=-1) for embs in (embs1, embs2))
            elif pool_type.startswith('mm'):
                embs1 = hidden_states[ex_id, start1:end1] # hidden
                embs2 = hidden_states[ex_id, start2:end2] # hidden
                last = '' if len(pool_type)<3 else pool_type[2]
                emb1, emb2 = (torch.cat([embs.mean(dim=0), embs.max(dim=0).values] + ([] if last=='' else [embs[0]] if last=='f' else [embs.min(dim=0).values]), dim=-1) for embs in (embs1, embs2))
            else:
                raise ValueError(f'wrong pool_type: {pool_type}')
            if merge_type == 'featwise_mul':
                merged_feature = emb1 * emb2 # hidden
            elif merge_type == 'diff':
                merged_feature = emb1 - emb2
            elif merge_type == 'concat':
                merged_feature = torch.cat((emb1, emb2)) # 2 * hidden
            elif merge_type == 'mulnorm':
                merged_feature = (emb1 / emb1.norm(dim=-1, keepdim=True)) * (emb2 / emb2.norm(dim=-1, keepdim=True))
            elif merge_type == 'comb_cm':
                merged_feature = torch.cat((emb1, emb2, emb1 * emb2))
            elif merge_type == 'comb_cmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1, emb2, emb1n*emb2n))
            elif merge_type == 'comb_cd':
                merged_feature = torch.cat((emb1, emb2, emb1 - emb2))
            elif merge_type == 'comb_cnmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1n, emb2n, emb1n* emb2n))
            elif merge_type == 'comb_dmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1 - emb2, emb1n* emb2n))
            elif merge_type == 'comb_dnmn':
                emb1n = emb1 / emb1.norm(dim=-1, keepdim=True)
                emb2n = emb2 / emb2.norm(dim=-1, keepdim=True)
                merged_feature = torch.cat((emb1n - emb2n, emb1n* emb2n))
            elif merge_type.startswith( 'dist_'):
                ops = re.findall(r'(l1|l2|dot)(n1?)?',merge_type)
                embs = {'': (emb1, emb2)} 
                norms = list(zip(*ops))[-1]
                if 'n' in norms:
                    embs['n'] = ( emb1 / emb1.norm(dim=-1, keepdim=True), emb2 / emb2.norm(dim=-1, keepdim=True) )
                if 'n1' in norms:
                    embs['n1'] = ( emb1 / emb1.norm(dim=-1, keepdim=True, p=1), emb2 / emb2.norm(dim=-1, keepdim=True, p=1) )
 
                dists = []
                for op, norm in ops:
                    e1, e2 = embs[norm]
                    if 'l1'==op:        
                        dists.append(torch.norm(e1-e2, dim=-1, p=1, keepdim=True))
                    elif 'l2'==op:        
                        dists.append(torch.norm(e1-e2, dim=-1, p=2, keepdim=True))
                    elif 'dot'==op:        
                        dists.append((e1*e2).sum(dim=-1, keepdim=True))
                    else:
                        raise ValueError('Unknown op={op} in merge_type={merge_type}')
                merged_feature = torch.cat(dists)

            features.append(merged_feature.unsqueeze(0))
        output = torch.cat(features, dim=0) # bs x hidden
        return output


    def convert_dataset_to_features(
            self, source_dir, logger
    ):
        examples = self.data_processor.get_examples(source_dir)
        if len(examples) == 0:
            print('ERROR: No examples found in the specified directory:', source_dir)
            raise ValueError(source_dir)
        return self.convert_examples_to_features(examples, logger)

    def convert_examples_to_features(self, examples, logger):
        features = []
        max_seq_len = self.local_config['max_seq_len']
#        import pdb; pdb.set_trace()
#        import pdb; pdb.set_trace()
        syns = self.local_config['syns']
        syn_label_to_id = {'T': 1, 'F': 0}
        syns_lab_to_pos = {lab: i for i, lab in enumerate(syns)}
        num_long_shortened, num_long_skipped = 0, 0

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
                positions[syns_lab_to_pos['Target'] * 2] = len(tokens)  # start position in subwords of the 1st occurrence
                target_subtokens = self.tokenizer.tokenize(target1)
                if self.local_config['mask_syns']:
                    tokens += [self.tokenizer.mask_token] * len(target_subtokens)
                else:
                    tokens += target_subtokens
                positions[syns_lab_to_pos['Target'] * 2 + 1] = len(tokens)  # end position in subwords of the 1st occurrence

                if right1:
                    tokens += self.tokenizer.tokenize(right1) 
                sep_pos = len(tokens)
                tokens += [self.tokenizer.sep_token]
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
                    tokens += self.tokenizer.tokenize(right2) 
                tokens += [self.tokenizer.sep_token]

                if len(tokens) > max_seq_len:
                    if True:
                        # New version of clipping to max_seq_len: does not skip examples, tries to balance
                        # the lengths of left and right contexts for each word usage
#                        import pdb; pdb.set_trace()
                        seg_pos = [positions[syns_lab_to_pos['Target'] * 2], sep_pos, positions[syns_lab_to_pos['Synonym'] * 2], len(tokens)]
                        seg_len = [seg_pos[0]] + [seg_pos[i+1]-seg_pos[i] for i in range(0,len(seg_pos)-1)]
                        good_seg_len = max_seq_len // len(seg_len)
                        #If all segments are of length at most good_seg_len, then everything shall fit. Thus, leave such segements and shorten those that are longer proportionally to their extra length.

                        extra_toks = [max(0, l-good_seg_len)  for l in seg_len]
                        remove_props = [e / sum(extra_toks) for e in extra_toks] # 0.0 for segments of at most good_seg_len, sum to 1.0
                        ll = len(tokens) - max_seq_len
                        remove_toks = [ceil(ll*p) for p in remove_props]
                        tokens = [tokens[0]] + tokens[1+remove_toks[0]:seg_pos[1]-remove_toks[1]] + [tokens[seg_pos[1]]] + tokens[seg_pos[1]+1+remove_toks[2]:seg_pos[3]-1-remove_toks[3]] + [tokens[seg_pos[3]-1]]
                        positions[syns_lab_to_pos['Target'] * 2] -= remove_toks[0]
                        positions[syns_lab_to_pos['Target'] * 2 + 1] -= remove_toks[0]
                        positions[syns_lab_to_pos['Synonym'] * 2] -= sum(remove_toks[0:-1])
                        positions[syns_lab_to_pos['Synonym'] * 2 + 1] -= sum(remove_toks[0:-1])
                        #print(tokens[slice(*positions[0:2])], tokens[slice(*positions[2:])]) 
                        num_long_shortened += 1
                    else:
                        # Old version of clipping to max_seq_len: skips some examples!
                        tokens = tokens[:max_seq_len]
                        if max(positions) > max_seq_len - 1:
                            print(f'Positions={positions} ({st1}-{end1}, {st2}-{end2}) are larger than max_seq_len={max_seq_len}: SKIPPING ')
                            num_long_skipped += 1
                            continue
                        else:
                            num_long_shortened += 1 
                    

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
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
                    assert self.local_config['loss'] in {'mseplus_loss','mse_loss'}, 'should be mse loss when training scd'
                    features.append(
                        WiCFeature2(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            syn_label=ex.score if ex.score != -1 else -syn_label_to_id[label],
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
        num_ex = len(examples)*2 if self.local_config['symmetric'] else len(examples)
        logger.info(f"Proportion of long examples shortened: {num_long_shortened/num_ex}, skipped: {num_long_skipped/num_ex}")
        return features
