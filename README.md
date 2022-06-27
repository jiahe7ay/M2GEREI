# M2GEREI

## Dataset
The expected structure of files is:

M2GEREI

  -- dataset
 
    -- docred
 
     -- train_annotated.json   
 
     -- train_distant.json
 
     -- dev.json
     
     -- test.json
 
   -- meta
 
    -- rel2id.json
 
 ## Training and Inference
 
 Train M2GEREI-BERT on DocRED with the following commands:

>> bash scripts/train_bert.sh M2GEREI test 
>> bash scripts/test_bert.sh M2GEREI test 

Alternatively, you can train M2GEREI-RoBERTa using:

>> bash scripts/train_roberta.sh M2GEREI test 
>> bash scripts/test_roberta.sh M2GEREI test 
