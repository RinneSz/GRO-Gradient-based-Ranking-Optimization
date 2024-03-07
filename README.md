# GRO-Gradient-based-Ranking-Optimization
 
Implementation for [Defense Against Model Extraction Attacks on Recommender Systems](http://arxiv.org/abs/2310.16335), WSDM24.

## Requirements
Python==3.7.16 and PyTorch==1.13.1.

## Updates
- Now supporting Amazon-Beauty dataset.
- Fixed several small issues in the code and the settings.

## Train a target model

1.To train a gold standard model without any poisoning:
```
python train_gd.py --device cuda:0 --rankall --dataset_code ml-1m --model_code bert --bb_model_code bert
```
- Note: to avoid any ambiguity when running other commands, please make sure that --model_code and --bb_model_code have 
the same input model name when running train_gd.py. For example, the above command is to train a gold standard Bert4Rec
on ml-1m. If you want to train a gold standard NARM, simply change both --model_code and --bb_model_code into narm.

2.To train a target model protected by GRO:
```
python train_gro.py --device cuda:0 --lamb 1 --rankall --dataset_code ml-1m --model_code bert --bb_model_code bert
python train_gro.py --device cuda:0 --lamb 0.01 --rankall --use_pretrained --dataset_code ml-20m --model_code bert --bb_model_code bert
python train_gro.py --device cuda:0 --lamb 0.01 --rankall --use_pretrained --dataset_code steam --model_code bert --bb_model_code bert
python train_gro.py --device cuda:0 --lamb 0.001 --rankall --use_pretrained --dataset_code beauty --model_code bert --bb_model_code bert
```
- Same as running train_gd.py, please parse the same model name into --model_code and --bb_model_code. If --use_pretrained is presented, it will load the corresponding gold standard model and continue to train it with GRO. Therefore, you need to have the corresponding gold standard model already trained. 
- The best Lambda is given above.
- Note that ml-1m does not necessarily need to use a pre-trained model because it runs fast.

## Conduct the extraction attack

1.To conduct the model extraction attack on the gold standard model:
```
python distill_gd.py --device cuda:0 --rankall --defense_mechanism reverse --dataset_code ml-1m --model_code bert --bb_model_code bert
```
- --defense_mechanism specifies the heuristic method (none, random, reverse) used to defend against model extraction.
--model_code specifies the architecture of the attacker's surrogate model. --bb_model_code specifies the model architecture
of the black-box target model. The above command uses a surrogate Bert4Rec to extract a gold standard Bert4Rec which is not
protected by the Reverse defense strategy. If you want to use a surrogate SASRec to extract the gold standard Bert4Rec, simply change
the --model_code into sas.

2.To conduct the model extraction attack on the model trained with GRO:
```
python distill_gro.py --device cuda:0 --lamb 0.01 --rankall --dataset_code ml-20m --model_code bert --bb_model_code bert --num_generated_seqs 3000
```
- It will use a surrogate model indicated by --model_code to extract the target model indicated by --bb_model_code. Please ensure the hyperparameter lambda here has the same value as you train GRO.

## Results
We report experiment results for four datasets (Amazon-Beauty, Steam, ML-1M, ML-20M) under different defense strategies (GRO, None, Random, Reverse). 
All numbers are in percentages. "Target" denotes the target model being protected by the corresponding defense strategy, while "Surrogate" denotes attacker's extracted model.

Amazon-Beauty:

|                      | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|----------------------|-------|-------|---------|---------|
| _GRO_ Target         | 2.85  | 4.48  | 1.44    | 1.85    | 
| _GRO_ Surrogate      | 2.26  | 3.58  | 1.15    | 1.48    |
| _None_ Target        | 2.90  | 4.67  | 1.46    | 1.88    |
| _None_ Surrogate     | 2.78  | 4.31  | 1.44    | 1.78    |
| _Random_ Target      | 1.19  | 2.30  | 0.54    | 0.82    |
| _Random_ Surrogate   | 1.26  | 2.51  | 0.56    | 0.87    |
| _Reverse_ Target     | 0.64  | 1.26  | 0.30    | 0.45    |
| _Reverse_ Surrogate  | 0.61  | 1.36  | 0.27    | 0.46    |

ML-1M:

|                      | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|----------------------|-------|-------|---------|---------|
| _GRO_ Target         | 18.08 | 28.96 | 9.53    | 12.48   | 
| _GRO_ Surrogate      | 11.46 | 21.94 | 5.19    | 7.82    |
| _None_ Target        | 20.18 | 30.70 | 10.61   | 13.26   |
| _None_ Surrogate     | 15.11 | 24.93 | 7.51    | 9.98    |
| _Random_ Target      | 5.84  | 12.08 | 2.70    | 4.26    |
| _Random_ Surrogate   | 11.61 | 20.59 | 5.25    | 7.50    |
| _Reverse_ Target     | 1.95  | 4.11  | 0.90    | 1.44    |
| _Reverse_ Surrogate  | 8.61  | 15.43 | 4.02    | 5.73    |

ML-20M:

|                      | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|----------------------|-------|-------|---------|---------|
| _GRO_ Target         | 13.95 | 21.63 | 7.23    | 9.02    | 
| _GRO_ Surrogate      | 8.03  | 14.42 | 3.70    | 5.30    |
| _None_ Target        | 14.86 | 22.50 | 7.77    | 9.67    |
| _None_ Surrogate     | 9.62  | 16.34 | 4.62    | 6.31    |
| _Random_ Target      | 4.64  | 9.45  | 2.12    | 3.32    |
| _Random_ Surrogate   | 6.38  | 11.67 | 2.90    | 4.22    |
| _Reverse_ Target     | 1.77  | 3.77  | 0.80    | 1.30    |
| _Reverse_ Surrogate  | 3.70  | 7.00  | 1.72    | 2.54    |

Steam:

|                      | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|----------------------|-------|-------|---------|---------|
| _GRO_ Target         | 19.87 | 24.69 | 15.39   | 16.58   | 
| _GRO_ Surrogate      | 19.05 | 23.50 | 14.96   | 16.08   |
| _None_ Target        | 19.93 | 24.84 | 15.43   | 16.69   |
| _None_ Surrogate     | 19.46 | 24.11 | 15.28   | 16.42   |
| _Random_ Target      | 4.37  | 8.77  | 1.99    | 3.09    |
| _Random_ Surrogate   | 15.70 | 20.75 | 10.67   | 11.94   |
| _Reverse_ Target     | 1.56  | 3.26  | 0.69    | 1.12    |
| _Reverse_ Surrogate  | 2.96  | 5.56  | 1.38    | 2.03    |



## Citation
If you find this repository helpful, please cite our paper:
```
@article{zhang2023defense,
  title={Defense Against Model Extraction Attacks on Recommender Systems},
  author={Zhang, Sixiao and Yin, Hongzhi and Chen, Hongxu and Long, Cheng},
  journal={arXiv preprint arXiv:2310.16335},
  year={2023}
}
```