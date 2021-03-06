# M2GEREI

## Requirements

- Python (tested on 3.7.4)

- CUDA (tested on 10.2)

- PyTorch (tested on 1.7.0)

- Transformers (tested on 3.4.0)

- numpy (tested on 1.19.4)

- opt-einsum (tested on 3.3.0)

- wandb

- ujson

- tqdm


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
## Results
dev results: ![](https://github.com/jiahe7ay/M2GEREI/blob/main/results_image/M2base_dev.png)

test results: ![](https://github.com/jiahe7ay/M2GEREI/blob/main/results_image/M2_TEST.png)

## Tips
If you want to get the relevant results of the test set, please log in to [codalab](https://codalab.lisn.upsaclay.fr/) to run online and get the results.
