# CodRED

Dataset and baseline code for the EMNLP 2021 paper [CodRED: A Cross-Document Relation Extraction Dataset for Acquiring Knowledge in the Wild](https://aclanthology.org/2021.emnlp-main.366/)

CodRED is the first human-annotated cross-document relation extraction (RE) dataset, aiming to test the RE systems’ ability of knowledge acquisition in the wild. CodRED has the following features: 
+ it requires natural language understanding in different granularity, including coarse-grained document retrieval, as well as fine-grained cross-document multi-hop reasoning; 
+ it contains 30,504 relational facts associated with 210,812 reasoning text paths, as well as enjoys a broad range of balanced relations, and long documents in diverse topics; 
+ it provides strong supervision about the reasoning text paths for predicting the relation, to help guide RE systems to perform meaningful and interpretable reasoning; 
+ it contains adversarially-created hard NA instances to avoid RE models to predict relations by inferring from entity names instead of text information.

The competition for evaluation on test set will be released soon.

### Baseline

Requirements:

```
pip install redis tqdm sklearn numpy
pip install transformers==4.3.3
pip install eveliver==1.21.0
```

<!-- 
apex need to be installed from GitHub:
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
``` 
-->


Then download the following files: `wiki_ent_link.jsonl`, `distant_documents.jsonl`, `popular_page_ent_link.jsonl` to `baseline/data/rawdata/`:

```
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/wiki_ent_link.jsonl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/distant_documents.jsonl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/popular_page_ent_link.jsonl
```

To run the baseline (Table 3 in the paper, closed setting, end-to-end model):

```
cd baseline/data/
python load_data_doc.py
python redis_doc.py
cd ../codred-blend
python -m torch.distributed.launch --nproc_per_node=4 codred-blend.py --train --dev --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 3e-5 --num_workers 2 --logging_step 10
```

<!-- At least 4 GPUs are required -->
<!-- 
If you encounter
```
venv/lib/python3.7/site-packages/eveliver/trainer.py", line 322, in load_data
     train_dataset, dev_dataset, test_dataset = self.callback.load_data()
ValueError: not enough values to unpack (expected 3, got 2)
```
Try to delete test_dataset. 
-->


The result is `AUC=46.69, F1=51.03`.
<!-- The final result is at the end of the r/7/output/dev-stat-1.json file. -->
<!-- The result is `AUC=48.59, F1=51.99`. -->

Arguments:

* `--positive_only`: Only use path with positive relations.
* `--positive_ep_only`: Only use entity pair with positive path.
* `--no_doc_pair_supervision`: Not use path-level supervision.
* `--no_additional_marker`: Not use additional marker `[UNUSEDx]`.
* `--mask_entity`: Use `[MASK]` to replace entity.
* `--single_path`: Randomly choose a path for every entity pair
* `--dsre_only`: use intra-document relation prediction, not use cross-document relation prediction.
* `--raw_only`: use cross-document relation prediction, not use intra-document relation prediction.

To run experiments with evidence sentence:

```
python -m torch.distributed.launch --nproc_per_node=4 codred-evidence.py --train --dev --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 3e-5 --num_workers 2 --logging_step 10
```

## Cite
If you use the dataset or the code, please cite this paper:
```
@inproceedings{yao-etal-2021-codred,
    title = "{C}od{RED}: A Cross-Document Relation Extraction Dataset for Acquiring Knowledge in the Wild",
    author = "Yao, Yuan and Du, Jiaju and Lin, Yankai and Li, Peng and Liu, Zhiyuan and Zhou, Jie and Sun, Maosong",
    booktitle = "Proceedings of EMNLP 2021",
    year = "2021",
    pages = "4452--4472",
}
```
