#!/bin/sh
#SBATCH --time=120:00:00
#SBATCH --gpus=titanx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000
#SBATCH --job-name prv_titan
#SBATCH --output prv_titan.out

direnv allow . && eval "\$(direnv export bash)"


dim='valence'

# RNN



# GENERAL
patience=10
n_seeds=5

csv='../baseline_results/stress/prelim/45_45/valence/valence.csv'



srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature egemaps --normalize --emo_dim "$dim" --model_type rnn --model_dim 256 --rnn_n_layers 4 --lr 0.002 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 200 --hop_len 100 --rnn_dropout 0.5 --cache

srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature ds --emo_dim "$dim" --model_type rnn --model_dim 64 --rnn_n_layers 2 --lr 0.001 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 100 --hop_len 50 --rnn_dropout 0. --cache

srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature w2v-msp --emo_dim "$dim" --model_type rnn --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 100 --hop_len 50 --rnn_dropout 0. --cache

srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature faus --emo_dim "$dim" --model_type rnn --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 200 --hop_len 100 --rnn_dropout 0. --cache

srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature vit --normalize --emo_dim "$dim" --model_type rnn --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.001 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 200 --hop_len 100 --rnn_dropout 0. --cache

srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature deepface --emo_dim "$dim" --model_type rnn --model_dim 128 --rnn_n_layers 2 --lr 0.005 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 200 --hop_len 100 --rnn_dropout 0. --cache

#srun python3 ../MuSe-2023-internal/main.py --task personalisation --feature biosignals --emo_dim "$dim" --model_type rnn --model_dim 256 --rnn_n_layers 4 --lr 0.005 --n_seeds 5 --seed 101 --use_gpu --result_csv "$csv" --win_len 200 --hop_len 100 --rnn_dropout 0. --cache
