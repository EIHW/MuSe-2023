# AROUSAL

# A
python3 personalisation.py --model_id RNN_2023-04-11-09-12_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] --eval_personalised 101_personalised_2023-04-11-11-34-52 --normalize --emo_dim physio-arousal

# V
python3 personalisation.py --model_id RNN_2023-04-11-09-41_[facenet]_[physio-arousal]_[256_1_False_64]_[0.001_256] --eval_personalised 101_personalised_2023-04-11-12-03-26 --emo_dim physio-arousal

# VALENCE

# A
python3 personalisation.py --model_id RNN_2023-04-11-09-11_[egemaps]_[valence]_[256_4_False_64]_[0.002_256] --eval_personalised 102_personalised_2023-04-11-14-36-31 --normalize --emo_dim valence

# V
python3 personalisation.py --model_id RNN_2023-04-11-09-34_[facenet]_[valence]_[128_2_False_64]_[0.005_256] --eval_personalised 104_personalised_2023-04-11-15-04-28 --emo_dim valence
