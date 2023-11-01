# GRO-Gradient-based-Ranking-Optimization

Official implementation of [Defense Against Model Extraction Attacks on Recommender Systems](http://arxiv.org/abs/2310.16335), WSDM24.

## Requirements
Python==3.7.16 and PyTorch==1.13.1.


## Train a target model

1.To train a gold standard model without any poisoning:
```
python train_gd.py --device cuda:0 --rankall --dataset_code ml-1m --model_code bert --bb_model_code bert
```
- Note: to avoid any ambiguity when running other commands, please make sure that --model_code and --bb_model_code have 
the same input model name when running train_gd.py. For example, the above command is to train a gold standard Bert4Rec
on ml-1m. If you want to train a gold standard NARM, simply change both --model_code and --bb_model_code into narm.

2.To train a target model with GRO:
```
python train_gro.py --device cuda:0 --lamb 1.0 --rankall --use_pretrained --dataset_code ml-20m --model_code bert --bb_model_code bert
```
- Same as running train_gd.py, please parse the same model name into --model_code and --bb_model_code. If --use_pretrained is presented, it will load the corresponding gold standard model and continue to train it with GRO as 
described in our paper. Therefore, you need to have the corresponding gold standard model already trained. 
- The best Lambda is given in our paper, as well as in utils.py.
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
python distill_gro.py --device cuda:0 --lamb 1.0 --rankall --dataset_code ml-20m --model_code bert --bb_model_code bert --num_generated_seqs 3000
```
- It will use a surrogate model indicated by --model_code to extract the target model indicated by --bb_model_code.
