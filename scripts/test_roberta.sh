# bash scripts/test_roberta.sh M2gerri test hoi

seed=66

ablation=$1
name=$2


load_path=chkpt/M2gerri_roberta_${ablation}_${name}_best.pt

evi_pred_file=evi_result_${ablation}_roberta-large.pkl

eval_mode=test

indicate_mode=indicate
indicate_ablation=evi_pred


evi_eval_mode=none

echo ablation ${ablation}
echo indicate ${indicate_mode}, ${indicate_ablation}
echo evi_eval_mode ${evi_eval_mode}

python train.py --data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed ${seed} \
--num_class 97 \
--save_path chkpt/M2gerri_roberta_${ablation}_${name}_test.pt \
--ablation ${ablation} \
--name ${name} \
--feature_path saved_features \
--train_sen_mode ${train_sen_mode} \
--indicate_mode ${indicate_mode} \
--indicate_ablation ${indicate_ablation} \
--eval_mode ${eval_mode} \
--evi_eval_mode ${evi_eval_mode} \
--evi_pred_file ${evi_pred_file} \
--load_path ${load_path} \
