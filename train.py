import sys
import os
import autoRun
autoRun.choose_gpu(retry=True, min_gpu_memory=10000, sleep_time=30)

import argparse

import numpy as np
import torch
import torch.optim as optim

# from apex import amp
import apex
import torch.cuda
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import * # to_official, official_evaluate, gen_official, to_score
import pickle
import copy

from tqdm import tqdm
from IPython import embed


def train(args, model, train_features,train_features2, dev_features, test_features, tokenizer=None):
    def finetune(features,features2 ,optimizer, num_epoch, tokenizer=None):
        cur_model = model.module if hasattr(model, 'module') else model
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_dataloader2 = DataLoader(features2, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        classfier=indicator=model
        for epoch in train_iterator:
            model.zero_grad()
            for step, data in enumerate( tqdm(zip(train_dataloader,train_dataloader2))):
                model.train()
                batch=data[0]
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }

                if args.ablation in ['M2GEREI']:
                    inputs['sen_labels'] = batch[5]
                    # print("sen ;abe;s")
                    # print(type(inputs['sen_labels'][0]))
                    inputs['sen_pos'] = batch[6]


                outputs = classfier(**inputs)
                sen_batch=data[1]
                inputs = {'input_ids': sen_batch[0].to(args.device),
                          'attention_mask': sen_batch[1].to(args.device),
                          'labels': sen_batch[2],
                          'entity_pos': sen_batch[3],
                          'hts': sen_batch[4],
                          }

                sent_outputs = indicator(**inputs)
                loss = outputs[0]+sent_outputs[0] / args.gradient_accumulation_steps
                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(cur_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if (step == 0 and epoch==0) or (step + 1) == len(train_dataloader) - 1:
                    print('epoch', epoch, "loss:", loss.item())
                    dev_score, dev_output, dev_pred, thresh = evaluate(args, model, dev_features, tag="dev")
                    print("classifer:")
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        with open("dev_result_"+ args.ablation + "_" + args.name + args.indicate_mode + '_' + args.model_name_or_path + ".json", "w") as fh:
                            json.dump(dev_pred, fh)

                        if test_features is not None:
                            pred = report(args, model, test_features)
                            with open("test_result_"+ args.ablation + "_" + args.name + args.indicate_mode + '_' + args.model_name_or_path + ".json", "w") as fh:
                                json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                    print('best f1', best_score)

    new_layer = ["extractor", "bilinear"]
    if args.ablation in ['M2GEREI']:
        new_layer.extend(['sr_bilinear'])

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.grouped_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
   # model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    set_seed(args)
    model.zero_grad()
    print('model initialized')
    finetune(train_features,train_features2, optimizer, args.num_train_epochs, tokenizer=tokenizer)

def evaluate(args, model, features,tag="dev", features2=None):
    sen_preds = []
    preds, scores, topks = [], [], []
    preds2, scores2, topks2 = [], [], []

    thresh = None

    if args.load_path != '' and args.indicate_mode != 'none':
        #dev文档中classfier_scores 有可能是保存好的，然后经过evaluate整个步骤后生成阈值返回给report函数来做test阶段，test阶段的classfier_scores
        #load_model = args.load_path.split('/')[1].split('.')[0]
        title2score_classfier = os.path.join(args.feature_path, 'title2score_classfier.pkl')
        title2score_indicator = os.path.join(args.feature_path, 'title2score_indicator.pkl')

        if os.path.exists(title2score_classfier) and os.path.exists(title2score_indicator) and args.indicate_mode == 'indicate':
            classfier_scores = pickle.load(open(title2score_classfier, 'rb'))
            indicator_scores = pickle.load(open(title2score_indicator, 'rb'))
            title2gt = extract_gt(args.feature_path, features)
            ans, thresh = indicate(classfier_scores, indicator_scores, title2gt, thresh=thresh)
            best_f1, best_evi_f1, best_f1_ign, _ = official_evaluate(ans, args.data_dir)

            output = {
                tag + "_F1": best_f1 * 100,
                tag + "_F1_ign": best_f1_ign * 100,
            }
            return best_f1, output, ans, thresh

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    print('Evaluating original', len(dataloader), 'samples...')
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    }
        if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
            inputs['sen_pos'] = batch[6]

        with torch.no_grad():
            if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
                inputs['return_senatt'] = True

                if args.indicate_mode != 'none':
                    inputs['return_score'] = True
                    pred, score, topk, sen_pred = model(**inputs)
                    score = score.cpu().numpy() # (bs, )
                    topk = topk.cpu().numpy()
                    scores.append(score)
                    topks.append(topk)
                else:
                    pred, sen_pred = model(**inputs)

                sen_pred = sen_pred.cpu().numpy()
                sen_pred[np.isnan(sen_pred)] = 0
                sen_preds.append(sen_pred)
            else:
                if args.indicate_mode != 'none':
                    inputs['return_score'] = True
                    pred, score, topk = model(**inputs)
                    score = score.cpu().numpy() # (bs, )
                    topk = topk.cpu().numpy()
                    scores.append(score)
                    topks.append(topk)
                else:
                    pred, *_ = model(**inputs)

            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    if args.indicate_mode != 'none':

        dataloader2 = DataLoader(features2, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
        print('indicator running:', len(features2), 'samples...')
        for batch in tqdm(dataloader2):
            model.eval()

            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'entity_pos': batch[3],
                      'hts': batch[4],
                      }

            with torch.no_grad():
                inputs['return_score'] = True
                pred2, score2, topk2 = model(**inputs)
                score2 = score2.cpu().numpy() # (bs, )
                topk2 = topk2.cpu().numpy()
                scores2.append(score2)
                topks2.append(topk2)

                pred2 = pred2.cpu().numpy()
                pred2[np.isnan(pred2)] = 0
                preds2.append(pred2)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    if len(sen_preds) > 0:
        sen_preds = np.concatenate(sen_preds, axis=0).astype(np.float32)

    if args.indicate_mode != 'none':
        print('Indicate mechanism in process!')
      #  preds2 = np.concatenate(preds2, axis=0).astype(np.float32)
        scores2 = np.concatenate(scores2, axis=0).astype(np.float32)
        topks2 = np.concatenate(topks2, axis=0).astype(int)
        indicator_scores = to_score(scores2, topks2, features2)

        scores = np.concatenate(scores, axis=0).astype(np.float32)
        topks = np.concatenate(topks, axis=0).astype(int)
        classfier_scores = to_score(scores, topks, features)

        title2gt = extract_gt(args.feature_path, features)
        ans, thresh = indicate(classfier_scores, indicator_scores, title2gt, thresh=thresh)

        if args.load_path != '':
            # save to file
            pickle.dump(classfier_scores, open(title2score_classfier, 'wb'))
            pickle.dump(indicator_scores, open(title2score_indicator, 'wb'))
    else:
        if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
            ans, evi_by_title = to_official(preds, features, sen_preds)
            # save predicted evidece to file
            output_evi_pred_file = "dev_evi_result_"+ args.ablation + "_" + args.model_name_or_path + ".pkl"
            if not os.path.exists(output_evi_pred_file):
                with open(output_evi_pred_file, "wb") as fh:
                    pickle.dump(evi_by_title, fh)
        else:
            ans = to_official(preds, features)

    if len(ans) > 0:
        best_f1, best_evi_f1, best_f1_ign, _ = official_evaluate(ans, args.data_dir, mode=tag)
    else:
        best_f1 = best_f1_ign = -1
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
        output[tag+'_evi_F1'] = best_evi_f1 * 100

    return best_f1, output, ans, thresh

def report(args, model, features, features2=None, thresh=None):
    sen_preds = []
    preds, scores, topks = [], [], []
    preds2, scores2, topks2 = [], [], []

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    print('Test: Evaluating original', len(dataloader), 'samples...')
    for batch in tqdm(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
            inputs['sen_pos'] = batch[6]

        with torch.no_grad():
            if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
                inputs['return_senatt'] = True

                if args.indicate_mode != 'none':
                    inputs['return_score'] = True
                    pred, score, topk, sen_pred = model(**inputs)
                    score = score.cpu().numpy() # (bs, )
                    topk = topk.cpu().numpy()
                    scores.append(score)
                    topks.append(topk)
                    preds.append(pred.cpu().numpy())
                else:
                    pred, sen_pred = model(**inputs)
                    preds.append(pred.cpu().numpy())
                sen_pred = sen_pred.cpu().numpy()
                sen_pred[np.isnan(sen_pred)] = 0
                sen_preds.append(sen_pred)
            else:
                if args.indicate_mode != 'none':
                    inputs['return_score'] = True
                    pred, score, topk = model(**inputs)
                    preds.append(pred.cpu().numpy())
                    score = score.cpu().numpy() # (bs, )
                    topk = topk.cpu().numpy()
                    scores.append(score)
                    topks.append(topk)
                else:
                    pred, *_ = model(**inputs)
                    preds.append(pred.cpu().numpy())

   # print(preds)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    if len(sen_preds) > 0:
        sen_preds = np.concatenate(sen_preds, axis=0).astype(np.float32)

    if args.indicate_mode != 'none':
        dataloader2 = DataLoader(features2, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
        print('indictor running:', len(dataloader2), 'samples...')
        for batch in tqdm(dataloader2):
            model.eval()

            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'entity_pos': batch[3],
                      'hts': batch[4],
                      }

            with torch.no_grad():
                inputs['return_score'] = True
                pred2, score2, topk2 = model(**inputs)
                score2 = score2.cpu().numpy() # (bs, )
                topk2 = topk2.cpu().numpy()
                scores2.append(score2)
                topks2.append(topk2)

                pred2 = pred2.cpu().numpy()
                pred2[np.isnan(pred2)] = 0
                preds2.append(pred2)

        print('Indicate mechanism in process!')
        scores = np.concatenate(scores, axis=0).astype(np.float32)
        topks = np.concatenate(topks, axis=0).astype(int)
        classfier_scores = to_score(scores, topks, features)

        preds2 = np.concatenate(preds2, axis=0).astype(np.float32)
        scores2 = np.concatenate(scores2, axis=0).astype(np.float32)
        topks2 = np.concatenate(topks2, axis=0).astype(int)
        indicator_scores = to_score(scores2, topks2, features2)

        title2gt = extract_gt(args.feature_path, features)
        ans, thresh= indicate(classfier_scores, indicator_scores, thresh=thresh)

    else:
        if args.ablation in ['M2GEREI'] and args.evi_eval_mode != 'none':
           # print(sen_preds)
            ans, evi_by_title = to_official(preds, features, sen_preds)
            # save predicted evidece to file
            output_evi_pred_file = "test_evi_result_"+ args.ablation + "_" + args.model_name_or_path + ".pkl"
            if not os.path.exists(output_evi_pred_file):
                with open(output_evi_pred_file, "wb") as fh:
                    pickle.dump(evi_by_title, fh)
        else:
            ans = to_official(preds, features)

    return ans
def generate_golden_envidence_text(args,features):
    h_idx, t_idx, title = [], [], []
    sent_labels = []
    for f in features:
        if 'original_hts' in f:
            hts = f['original_hts']
        else:
            hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sent_labels += [f["sen_labels"] for ht in hts]
        if 'htbs' in f:
            htbs = f['htbs']
            h_idx += [ht[0][0] for ht in htbs]
            t_idx += [ht[0][1] for ht in htbs]
            title += [f["title"] for ht in htbs]
            sent_labels += [f["sen_labels"] for ht in htbs]
    evi_by_title = {}
    num_pairs_with_evidence = 0
    for i in range(len(sent_labels)):
        sen = np.nonzero(sent_labels[i])[0].tolist()

                #print(sen_preds[i])

           # print(sen_pred)

        if len(sen) > 0:
            h,t,tit = h_idx[i], t_idx[i], title[i]
            if tit not in evi_by_title:
                evi_by_title[tit] = {}
            evi_by_title[tit][(h,t)] = sen
            num_pairs_with_evidence += 1
    output_evi_pred_file = "train_annotated_evi_result_" + args.ablation + '_' + args.model_name_or_path + ".pkl"
    if not os.path.exists(output_evi_pred_file):
        with open(output_evi_pred_file, "wb") as fh:
            pickle.dump(evi_by_title, fh)
def main():
    parser = argparse.ArgumentParser()

    if True:
        parser.add_argument("--ablation", default="atlop", type=str, choices=['atlop', 'M2GEREI'])
        parser.add_argument("--name", default="", type=str)
        parser.add_argument("--indicate_mode", default='none', choices=['none', 'indicate'])
        parser.add_argument("--indicate_ablation", default="", type=str)
        parser.add_argument("--eval_mode", default='dev_only', type=str)
        parser.add_argument("--evi_eval_mode", default='none', type=str)
        parser.add_argument("--max_sen_num", type=int, default=25)

        parser.add_argument("--data_dir", default="dataset/docred", type=str)
        parser.add_argument("--transformer_type", default="bert", type=str)
        parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

        parser.add_argument("--train_file", default="train_annotated.json", type=str)
        parser.add_argument("--dev_file", default="dev.json", type=str)
        parser.add_argument("--test_file", default="test.json", type=str)
        parser.add_argument("--rel2htype_file", default="rel2htype.json", type=str)

        parser.add_argument("--save_path", default="", type=str)
        parser.add_argument("--load_path", default="", type=str)
        parser.add_argument("--feature_path", default="", type=str)
        parser.add_argument("--evi_pred_file", default='', type=str)

        parser.add_argument("--config_name", default="", type=str,
                            			help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            			help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--max_seq_length", default=1024, type=int,
                            			help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")

        parser.add_argument("--train_batch_size", default=4, type=int,
                            			help="Batch size for training.")
        parser.add_argument("--test_batch_size", default=8, type=int,
                            			help="Batch size for testing.")
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                            			help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--num_labels", default=4, type=int,
                            			help="Max number of labels in prediction.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            			help="The initial learning rate for transformer layers for Adam.")
        parser.add_argument("--grouped_learning_rate", default=1e-4, type=float,
                            			help="The initial learning rate for new layers for Adam.")
        parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                            			help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            			help="Max gradient norm.")
        parser.add_argument("--warmup_ratio", default=0.06, type=float,
                            			help="Warm up ratio for Adam.")
        parser.add_argument("--num_train_epochs", default=30.0, type=float,
                            			help="Total number of training epochs to perform.")
        parser.add_argument("--evaluation_steps", default=-1, type=int,
                            			help="Number of training steps between evaluations.")
        parser.add_argument("--seed", type=int, default=66,
                            			help="random seed for initialization")
        parser.add_argument("--num_class", type=int, default=97,
                            			help="Number of relation types in dataset.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.save_path != "":
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)

    if args.load_path == '':
        train_features = read(args, train_file, tokenizer, if_inference=False)
        generate_golden_envidence_text(args, train_features)
        train_features2 = read(args, train_file, tokenizer, ablation=args.indicate_ablation)
    dev_features = read(args, dev_file, tokenizer)
    # dev_only为只用验证集不要测试集
    if args.eval_mode == 'dev_only':
        test_features = None
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(args, test_file, tokenizer)  # , max_sent_num=3053)

    test_features2 = dev_features2 = None
    # if args.indicate_mode != 'none':
    #     dev_features2 = read(args, dev_file, tokenizer, ablation=args.indicate_ablation)
    #
    #     if args.eval_mode != 'dev_only':
    #         test_features2 = read(args, test_file, tokenizer, ablation=args.indicate_ablation)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    if 'M2GEREI' in args.load_path:
        args.ablation = 'M2GEREI'

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels, ablation=args.ablation, max_sen_num=args.max_sen_num)
    model.to(device)
    # device_ids = [0, 1]
    if args.load_path != '':
        name = args.load_path.split('/')[1].split('.')[0].split('M2GEREI_')[1]
        print('Loading from', args.load_path)
        # model = apex.amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))

        if args.indicate_mode != 'none':
            args.indicate_mode = 'none'
            dev_score, dev_output, dev_pred, thresh = evaluate(args, model, dev_features)
            dev_features2 = read(args, dev_file, tokenizer, ablation=args.indicate_ablation)
            args.indicate_mode = 'indicate'
            dev_score, dev_output, dev_pred, thresh = evaluate(args, model, dev_features, tag="dev",
                                                               features2=dev_features2)


        else:
            dev_score, dev_output, dev_pred, thresh = evaluate(args, model, dev_features)
        print(dev_output)

        with open("dev_result_" + args.name + '_' + name + '_' + args.model_name_or_path + ".json", "w") as fh:
            json.dump(dev_pred, fh)

        if test_features is not None:
            print("testing!!!!!!!!!!!!!")
            if (args.indicate_mode != 'none'):
                args.indicate_mode = 'none'
                pred = report(args, model, test_features, features2=test_features2, thresh=thresh)
                args.indicate_mode = 'indicate'
                test_features2 = read(args, test_file, tokenizer, ablation=args.indicate_ablation)
                pred = report(args, model, test_features, features2=test_features2, thresh=thresh)
            else:
                pred = report(args, model, test_features, features2=test_features2, thresh=thresh)
            with open("test_result_" + args.name + '_' + name + '_' + args.model_name_or_path + ".json", "w") as fh:
                json.dump(pred, fh)
    else:
        train(args, model, train_features,train_features2, dev_features, test_features, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
