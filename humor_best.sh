python3 main.py --task humor --feature egemaps --normalize --model_dim 32 --rnn_n_layers 2 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5

python3 main.py --task humor --feature ds --model_dim 256 --rnn_n_layers 1 --lr 0.001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0

python3 main.py --task humor --feature w2v-msp --model_dim 128 --rnn_n_layers 2 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0

python3 main.py --task humor --feature bert-multilingual --model_dim 128 --rnn_n_layers 4 --lr 0.001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0

python3 main.py --task humor --feature faus --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5

python3 main.py --task humor --feature vit --normalize --model_dim 64 --rnn_n_layers 2 --lr 0.0001 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0

python3 main.py --task humor --feature facenet --model_dim 64 --rnn_n_layers 4 --lr 0.005 --early_stopping_patience 3 --reduce_lr_patience 1 --rnn_dropout 0.5