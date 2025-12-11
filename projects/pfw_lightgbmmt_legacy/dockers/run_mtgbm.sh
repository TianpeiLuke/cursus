#!/bin/bash

source activate pytorch_p38

# train and evaluate MultiTab model
python multitask_mtgbm_train_eval.py --exp_id final_mtgbm_abuse_features