### AROUSAL

#egemaps
python3 main.py --task personalisation --feature egemaps --normalize --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 4 --lr 0.002  --win_len 50 --hop_len 25 --rnn_dropout 0.5

# deepspectrum
python3 main.py --task personalisation --feature ds --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 2 --lr 0.005  --win_len 200 --hop_len 100 --rnn_dropout 0.

# w2v
python3 main.py --task personalisation --feature w2v-msp --emo_dim physio-arousal --model_dim 32 --rnn_n_layers 4 --rnn_bi --lr 0.005  --win_len 200 --hop_len 100 --rnn_dropout 0.5

# faus
python3 main.py --task personalisation --feature faus --emo_dim physio-arousal --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  --win_len 200 --hop_len 100 --rnn_dropout 0.

# vit
python3 main.py --task personalisation --feature vit --normalize --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 4 --rnn_bi --lr 0.005  --win_len 50 --hop_len 25 --rnn_dropout 0.5

# facenet
python3 main.py --task personalisation --feature facenet --emo_dim physio-arousal --model_dim 256 --rnn_n_layers 1 --lr 0.001  --win_len 50 --hop_len 25 --rnn_dropout 0.5


### VALENCE

# egemaps
python3 main.py --task personalisation --feature egemaps --normalize --emo_dim valence --model_dim 256 --rnn_n_layers 4 --lr 0.002  --win_len 200 --hop_len 100 --rnn_dropout 0.5

# deepspectrum
python3 main.py --task personalisation --feature ds --emo_dim valence --model_dim 64 --rnn_n_layers 2 --lr 0.001  --win_len 100 --hop_len 50 --rnn_dropout 0.

# w2v-msp
python3 main.py --task personalisation --feature w2v-msp --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  --win_len 100 --hop_len 50 --rnn_dropout 0.

# faus
python3 main.py --task personalisation --feature faus --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.005  --win_len 200 --hop_len 100 --rnn_dropout 0.

# vit
python3 main.py --task personalisation --feature vit --normalize --emo_dim valence --model_dim 128 --rnn_n_layers 4 --rnn_bi --lr 0.001  --win_len 200 --hop_len 100 --rnn_dropout 0.

# facenet
python3 main.py --task personalisation --feature facenet --emo_dim valence --model_dim 128 --rnn_n_layers 2 --lr 0.005  --win_len 200 --hop_len 100 --rnn_dropout 0.
