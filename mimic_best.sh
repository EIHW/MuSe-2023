python3 main.py --task mimic --feature egemaps --normalize --model_type rnn --model_dim 256 --rnn_n_layers 2 --lr 0.001 --rnn_dropout 0.5 --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature deepspectrum --model_type rnn --model_dim 256 --rnn_n_layers 4 --lr 0.0005 --rnn_dropout 0.5 --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature w2v-msp --model_type rnn --model_dim 128 --rnn_n_layers 2 --lr 0.001 --rnn_dropout 0.5 --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature faus --model_type rnn --model_dim 256 --rnn_n_layers 4 --lr 0.0005 --rnn_bi --rnn_dropout 0. --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature electra --model_type rnn --model_dim 128 --rnn_n_layers 1 --rnn_bi --lr 0.005 --rnn_dropout 0. --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature vit --normalize --model_type rnn --model_dim 256 --rnn_n_layers 4 --rnn_bi --lr 0.001 --rnn_dropout 0.5 --early_stopping_patience 10 --reduce_lr_patience 5

python3 main.py --task mimic --feature facenet --model_type rnn --model_dim 32 --rnn_n_layers 1 --lr 0.005 --rnn_dropout 0. --early_stopping_patience 10 --reduce_lr_patience 5
