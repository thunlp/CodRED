# CODRED

### Baseline

Requirements:

```
pip install redis tqdm sklearn numpy
pip install transformers==4.3.3
pip install eveliver==1.21.0
```

Then download the following files: `wiki_ent_link.jsonl`, `distant_documents.jsonl`, `popular_page_ent_link.jsonl` to `baseline/data/rawdata/` from Aliyun OSS.

To run the baseline (Table 3, closed, end-to-end):

```
cd baseline/data/
python load_data_doc.py
python redis_doc.py
cd ../codred-blend
python -m torch.distributed.launch --nproc_per_node=4 codred-blend.py --train --dev --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 3e-5 --num_workers 2 --logging_step 10
```

The result is `AUC=46.69, F1=51.03`.

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