# M2GEREI

## Dataset
The DocRED dataset can be downloaded following the instructions at  [link](https://drive.google.com/drive/folders/1owp7ZRbrMl_s1ljIh6AvnmniLJSliV6h).

The expected structure of files is:
```
M2GEREI
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- meta
 |    |-- rel2id.json

```
 ## Training and Inference
 
 Train M2GEREI-BERT on DocRED with the following commands:
``` 
bash scripts/train_bert.sh M2GEREI test 

bash scripts/test_bert.sh M2GEREI test 
``` 
Alternatively, you can train M2GEREI-RoBERTa using:

``` 
bash scripts/train_roberta.sh M2GEREI test 

bash scripts/test_roberta.sh M2GEREI test 
``` 
