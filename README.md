# M2GEREI

## Dataset
The expected structure of files is:

M2GEREI

 |-- coref_results
 
 |    |-- train_annotated_coref_results.json
 
 |    |-- dev_coref_results.json
 
 |    |-- test_coref_results.json
 
 ## Training and Inference
 
 Train M2GEREI-BERT on DocRED with the following commands:

>> bash scripts/train_bert.sh M2GEREI test 
>> bash scripts/test_bert.sh M2GEREI test 
Alternatively, you can train M2GEREI-RoBERTa using:

>> bash scripts/train_roberta.sh M2GEREI test 
>> bash scripts/test_roberta.sh M2GEREI test 
