#### --model_id and --checkpoint_seed must be adapted accordingly

### AROUSAL

# egemaps
python3 personalisation.py --model_id RNN_2023-04-11-09-12_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] --normalize --checkpoint_seed 101 --emo_dim physio-arousal --lr 0.002 --early_stopping_patience 10 --epochs 20 --win_len 10 --hop_len 5

# ds
python3 personalisation.py --model_id RNN_2023-04-11-09-16_[ds]_[physio-arousal]_[32_2_False_64]_[0.005_256] --checkpoint_seed 101 --emo_dim physio-arousal --lr 0.005 --early_stopping_patience 10 --epochs 20 --win_len 20 --hop_len 10

# w2v
python3 personalisation.py --model_id RNN_2023-04-11-09-23_[w2v-msp]_[physio-arousal]_[32_4_True_64]_[0.005_256] --checkpoint_seed 104 --emo_dim physio-arousal --lr 0.01 --early_stopping_patience 10 --epochs 10 --win_len 20 --hop_len 10

# faus
python3 personalisation.py --model_id RNN_2023-04-11-09-31_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] --checkpoint_seed 103 --emo_dim physio-arousal --lr 0.002 --early_stopping_patience 10 --epochs 5 --win_len 20 --hop_len 10

# vit 
python3 personalisation.py --model_id RNN_2023-04-11-09-35_[vit]_[physio-arousal]_[256_4_True_64]_[0.005_256] --normalize --checkpoint_seed 101 --emo_dim physio-arousal --lr 0.002 --early_stopping_patience 10 --epochs 5 --win_len 10 --hop_len 5

# facenet 
python3 personalisation.py --model_id RNN_2023-04-11-09-41_[facenet]_[physio-arousal]_[256_1_False_64]_[0.001_256] --checkpoint_seed 101 --emo_dim physio-arousal --lr 0.001 --early_stopping_patience 10 --epochs 50 --win_len 10 --hop_len 5


### VALENCE 

# egemaps
python3 personalisation.py --model_id RNN_2023-04-11-09-11_[egemaps]_[valence]_[256_4_False_64]_[0.002_256] --normalize --checkpoint_seed 102 --emo_dim valence --lr 0.001 --early_stopping_patience 10 --epochs 5 --win_len 20 --hop_len 10

# ds
python3 personalisation.py --model_id RNN_2023-04-11-09-15_[ds]_[valence]_[64_2_False_64]_[0.001_256] --checkpoint_seed 101 --emo_dim valence --lr 0.001 --early_stopping_patience 10 --epochs 10 --win_len 20 --hop_len 10

# w2v
python3 personalisation.py --model_id RNN_2023-04-11-09-18_[w2v-msp]_[valence]_[128_4_True_64]_[0.005_256] --checkpoint_seed 101 --emo_dim valence --lr 0.002 --early_stopping_patience 10 --epochs 20 --win_len 20 --hop_len 10

# faus
python3 personalisation.py --model_id RNN_2023-04-11-09-25_[faus]_[valence]_[128_4_True_64]_[0.005_256] --checkpoint_seed 103 --emo_dim valence --lr 0.001 --early_stopping_patience 10 --epochs 10 --win_len 20 --hop_len 10

# vit 
python3 personalisation.py --model_id RNN_2023-04-11-09-28_[vit]_[valence]_[128_4_True_64]_[0.001_256] --normalize --checkpoint_seed 101 --emo_dim valence --lr 0.001 --early_stopping_patience 10 --epochs 50 --win_len 10 --hop_len 5

# facenet 
python3 personalisation.py --model_id RNN_2023-04-11-09-34_[facenet]_[valence]_[128_2_False_64]_[0.005_256] --checkpoint_seed 104 --emo_dim valence --lr 0.002 --early_stopping_patience 10 --epochs 50 --win_len 10 --hop_len 5
