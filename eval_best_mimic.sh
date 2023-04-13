# best audio
python3 main.py --task mimic --eval_model RNN_2023-04-08-09-32_[w2v-msp]_[128_2_False_64]_[0.001_256] --feature w2v-msp --eval_seed 103 --predict

# best video
python3 main.py --task mimic --eval_model RNN_2023-04-08-09-57_[faus]_[256_4_True_64]_[0.0005_256] --feature faus --eval_seed 104 --predict

# best text
python3 main.py --task mimic --eval_model RNN_2023-04-08-10-08_[electra]_[128_1_True_64]_[0.005_256] --feature electra --eval_seed 102 --predict
