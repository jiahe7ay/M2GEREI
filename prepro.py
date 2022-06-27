from tqdm import tqdm
import ujson as json

import numpy as np
import random
from IPython import embed

from collections import defaultdict, Counter
import copy
import pickle
import os

from IPython import embed

dataset_path = 'dataset/'
docred_rel2id = json.load(open(dataset_path + 'meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in docred_rel2id.items()}
rel2name = json.load(open(dataset_path + 'meta/rel_info.json', 'r')) # PXX -> name
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

INF = 1e8

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res



def add_entity_markers(entities, original_sents, tokenizer):

    cls_token = tokenizer.special_tokens_map['cls_token']
    # if cls: add a sep token before every sentence

    entity_start, entity_end = [], []
    token2type = [['-'] * len(s) for s in original_sents ]
    for entity in entities:
        for mention in entity:
            sent_id = mention["sent_id"]
            pos = mention["pos"]
            entity_start.append((sent_id, pos[0],))
            entity_end.append((sent_id, pos[1] - 1,))
            token2type[sent_id][pos[0]] = mention['type']

    # new_map: old_pos (local) -> new_pos (global)
    # sent_map_local: list of new_map for each sent
    sents = []
    sent_map = []
    sents_local = []
    sent_map_local = []

    sen_starts, sen_ends = [], []

    for i_s, sent in enumerate(original_sents):
        new_map = {}
        sent_local = []
        new_map_local = {}

        sen_starts.append(len(sents))

        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token) # return list(str)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

            new_map_local[i_t] = len(sent_local)
            sent_local.extend(tokens_wordpiece)
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

        new_map_local[i_t + 1] = len(sents_local)
        sent_map_local.append(new_map_local)
        sents_local.append(sent_local)

        sen_ends.append(len(sents))

    sen_pos = [(sen_starts[i],sen_ends[i]) for i in range(len(sen_starts))]

    return sents, sent_map, sents_local, sent_map_local, sen_pos

def return_feature(tmp_text, entity_pos, relations, hts, title, tokenizer, original_hts=None, sen_labels=None, sen_pos=None, max_seq_length=1024):
    tmp_text = tmp_text[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tmp_text) # Returns the vocabulary as a dict of {token: index} pairs
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids) # build model inputs by concatenating and adding special tokens.

    feature = {'input_ids': input_ids, # sents converted by the tokenizer
               'entity_pos': entity_pos, # the [START, END] of each mention of each entity
               'labels': relations, # a list of relations of a pair, each is a one-hot vector
               'hts': hts, # a list of ([h, t]) pairs
               'title': title,
               }

    if original_hts is not None and len(original_hts) > 0:
        feature['original_hts'] = original_hts

    if sen_pos is not None and len(sen_pos) > 0:
        num_pos = np.sum(np.array(relations)[:,1:].sum(axis=-1) > 0)
        assert(num_pos == len(sen_labels))
        feature['sen_labels'] = sen_labels
        feature['sen_pos'] = sen_pos
    return feature

# global pos to local pos. pos in the form on [start, end]
def find_local_pos(sents, pos, sent_id=None):
    if sent_id is None:
        cur_length = 0
        for i,s in enumerate(sents):
            if pos[0] < cur_length + len(s):
                sent_id = i
                local_start = pos[0] - cur_length
                local_end = pos[1] - cur_length
                return sent_id, (local_start, local_end)
            cur_length += len(s)
    else:
        cur_length = sum([len(s) for s in sents[:sent_id]])
        local_start = pos[0] - cur_length
        local_end = pos[1] - cur_length
        return sent_id, (local_start, local_end)


def get_surface_name(sents, m):
    name = []
   # print(m)
    for i in range(m['pos'][0], m['pos'][1]):
        name.append(sents[m['sent_id']][i])
    return name

def pseudo_doc2feature(args, ablation, title, evidence, sents_local, entities, sent_map_local, train_triple, tokenizer):
    relations, hts = [], []
    original_hts = []

    pos, neg = 0, 0

    tmp_text = []
    for i in evidence:
        tmp_text.extend(sents_local[i])

    tmp_eids = []
    entity_pos = []
    for eid, e in enumerate(entities):
        e_poss = []
        for m in e:
            if m['sent_id'] not in evidence:
                continue
            offset = sum([len(sents_local[i]) for i in evidence if i<m['sent_id']]) # local_pos + len(previous sents in evidence)
            start = sent_map_local[m["sent_id"]][m["pos"][0]] + offset
            end = sent_map_local[m["sent_id"]][m["pos"][1]] + offset
            e_poss.append((start, end,))

        if len(e_poss) > 0: # if the entity has at least one mention that occurs in evidence
            entity_pos.append(e_poss)
            tmp_eids.append(eid)

    ht2hts_idx = {}
    for new_h, h0 in enumerate(tmp_eids):
        for new_t, t0 in enumerate(tmp_eids):
            if h0 == t0:
                continue

            relation = [0] * len(docred_rel2id)
            if (h0, t0) in train_triple:
                for m in train_triple[h0, t0]:
                    relation[m["relation"]] = 1

            if sum(relation) > 0:
                relations.append(relation)
                ht2hts_idx[(h0,t0)] = len(hts)
                hts.append([new_h, new_t])
                original_hts.append([h0, t0])
                pos += 1
            else:
                relation = [1] + [0] * (len(docred_rel2id) - 1)
                relations.append(relation)
                ht2hts_idx[(h0,t0)] = len(hts)
                hts.append([new_h, new_t])
                original_hts.append([h0, t0])
                neg += 1

    assert( np.all(np.array([len(r) for r in relations]) == 97))
    assert(len(relations) == len(hts))
    # print(len(relations), len(tmp_eids)*(len(tmp_eids) - 1) )
    # assert len(relations) == len(tmp_eids) * (len(tmp_eids) - 1)

    feature = return_feature(tmp_text, entity_pos, relations, hts, title, tokenizer, original_hts=original_hts)

    return feature, pos, neg

# @profile
def read_docred(args, file_in, tokenizer, ablation=None, if_inference=True):

    if args is not None:
        max_seq_length = args.max_seq_length
        feature_path = args.feature_path
        rel2htype_file = args.rel2htype_file
        max_sen_num = args.max_sen_num

    if ablation is None:
        ablation = args.ablation
  #  print(args.evi_pred_file)
    mode = file_in.split('/')[-1].split('.')[0]
    print(mode)
    print(args.evi_pred_file)
    evi_pred_file = mode + '_' + args.evi_pred_file
    print("file")
    print(evi_pred_file)
#./dataset/docred/train_annotated.json
    if feature_path != '':
        ablation_or_emarker = ablation

        feature_file = os.path.join(feature_path, ablation_or_emarker + '_' + mode )


        if args.transformer_type == 'roberta':
            feature_file = feature_file + '_roberta'

        feature_file = feature_file + '.pkl'

        if os.path.exists(feature_file):
            print('Feature file:', feature_file)
            features = pickle.load( open( feature_file, "rb") )
            print('Feature loaded from', feature_file)

            return features

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    count = 0
    coref_count = 0
    features = []
    evidence_lengths = []
    if file_in == "":
        return None
    print('Reading from:', file_in)
    with open(file_in, "r") as fh:
        data = json.load(fh)


    # print("ablation")
    # print(ablation)
    # M2gerri

    if ablation in ['evi_pred']:
        evi_pred_by_doc = pickle.load(open( evi_pred_file, "rb") )

    for doc_id, sample in tqdm( enumerate(data), desc="Example", total=len(data)): # each doc

        # add coref to entities

        entities_w_coref = entities = sample['vertexSet']

        eid2sent_id = {}
        for eid, e in enumerate( entities ):
            eid2sent_id[eid] = sorted(set([m['sent_id'] for m in e]))


        for ei in range(len(entities)):

            for ej in range(len(entities)):
                if ei == ej:
                    continue


        sents, sent_map, sents_local, sent_map_local, sen_pos = add_entity_markers(entities, sample['sents'], tokenizer)

        if ablation not in ['M2gerri']:
            sen_pos = []

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})


        if ablation in ['evi_pred'] and if_inference:
            sents_set = []
            for h in range(len(entities)):
                for t in range(len(entities)):
                    # produce tmp sent
                    if h == t:
                        continue

                    if ablation == 'evi_pred':
                        if sample['title'] not in evi_pred_by_doc or (h,t) not in evi_pred_by_doc[ sample['title'] ]:
                            continue
                        evidence = evi_pred_by_doc[ sample['title'] ][(h,t)]
                        evidence = [x for x in evidence if x < len(sample['sents']) ]



                    # check if h and t are both in evidence:
                    ts = eid2sent_id[t]
                    if len(set(ts).intersection(set(evidence))) == 0:
                        evidence.insert(0, ts[0])

                    hs = eid2sent_id[h]
                    if len(set(hs).intersection(set(evidence))) == 0:
                        evidence.insert(0, hs[0])

                    evidence = sorted(set(evidence))

                    if evidence in sents_set:
                        continue
                    sents_set.append(evidence)
                    evidence_lengths.append(len(evidence))

                    feature, pos, neg = pseudo_doc2feature(args, ablation, sample['title'], evidence, sents_local, entities, sent_map_local, train_triple, tokenizer)
                    pos_samples += pos
                    neg_samples += neg

                    i_line += 1
                    features.append(feature)

                    count += 1

        # print("abla")
        # print(ablation)
        #M2gerri
        if ablation not in ['evi_pred'] or not if_inference: # in training, add the whole document anyway..
            entity_pos = []
            for e in entities:
                entity_pos.append([])
                for m in e:
                    # print(m)
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    # print(start)
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))

            relations, hts = [], []
            evis, sen_evis = [], []
            ht2hts_idx = {}
            max_bridge_num = 5
            r_count=0
            s_count=0
            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h == t:
                        continue
                    if (h,t) in train_triple:
                        relation = [0] * len(docred_rel2id)
                        sen_evi = np.zeros( max_sen_num, dtype=int)
                        for mention in train_triple[h, t]:
                            relation[mention["relation"]] = 1
                        if ablation == 'M2gerri':
                            # print("yes")
                            for mention in train_triple[h, t]:
                                for i in mention['evidence']:
                                    sen_evi[i] = 1


                            ts = eid2sent_id[t]
                            if len(set(ts).intersection(set(evidence))) == 0:
                                evidence.insert(0, ts[0])

                            hs = eid2sent_id[h]
                            if len(set(hs).intersection(set(evidence))) == 0:
                                evidence.insert(0, hs[0])

                            for i in set(evidence):
                                sen_evi[i] = 1

                        if ablation in ['M2gerri']:
                            sen_evis.append(sen_evi)


                        relations.append(relation)
                        # print(len(relation))
                        # print(len(sen_evi))
                        ht2hts_idx[(h,t)] = len(hts)
                        hts.append([h, t])
                        pos_samples += 1
                    else:
                        relation = [1] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        ht2hts_idx[(h,t)] = len(hts)
                        hts.append([h, t])
                        neg_samples += 1

            # print("------------------")
            # print(len(hts))
            # print(len(sen_evis))
            feature = return_feature(sents, entity_pos, relations, hts, sample['title'], tokenizer, \
                                        sen_labels=sen_evis, \
                                        sen_pos=sen_pos)

            i_line += 1
            features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples)) # all entity pairs without any relations
    print('# of coref detected {}'.format(coref_count))
    if len(evidence_lengths) > 0:
        print('average sent num in evidence {}'.format(np.mean(evidence_lengths)))


    if feature_path != '':
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
            
        if not os.path.exists(feature_file):
            print('Saving to', feature_file)
            pickle.dump(features, open( feature_file, "wb") )

    return features