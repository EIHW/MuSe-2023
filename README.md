# MuSe-2023 Baseline Model: GRU Regressor


[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/370100318_The_MuSe_2023_Multimodal_Sentiment_Analysis_Challenge_Mimicked_Emotions_Cross-Cultural_Humour_and_Personalisation)


## Sub-challenges and Results 
For details, please see the [Baseline Paper](https://www.researchgate.net/publication/370100318_The_MuSe_2023_Multimodal_Sentiment_Analysis_Challenge_Mimicked_Emotions_Cross-Cultural_Humour_and_Personalisation). If you want to sign up for the challenge, please fill the form 
[here](https://www.muse-challenge.org/challenge/participate) for MuSe-Humor and MuSe-Stress. For MuSe-Reaction, please contact competitions \[at\] hume.ai 

* MuSe-Mimic: predicting the intensity of three emotions (Approval, Disappointment, Uncertainty). 
 *Official baseline* : **.4727** mean Pearson's correlation over all three classes.

* MuSe-Humor: predicting presence/absence of humour in cross-cultural (German/English) football press conference recordings. 
*Official baseline*: **.8310** AUC.

* MuSe-Personalisation: regression on valence and arousal signals for persons in a stressed disposition. In order to
facilitate personalisation, parts of the test subjects' labels are provided as well. *Official baselines*:
**.7827** CCC for valence, **.7482** CCC for (physiological) arousal, **.7639** CCC as the mean of best CCC for arousal and 
best CCC for valence (*Combined*). Note that the *Combined* score will be used to determine the challenge winner.

## Installation
It is highly recommended to run everything in a Python virtual environment. Please make sure to install the packages listed 
in ``requirements.txt`` and adjust the paths in `config.py` (especially ``BASE_PATH``). 

You can then e.g. run the unimodal baseline reproduction calls in the ``*.sh`` file provided for each sub-challenge.

## Settings
The ``main.py`` script is used for training and evaluating models for MuSe-Mimic, MuSe-Humor and the first step of the 
personalisation method applied for MuSe-Personalisation (cf. baseline paper).  Most important options:
* ``--task``: choose either `humor`, `mimic` or `personalisation` 
* ``--feature``: choose a feature set provided in the data (in the ``PATH_TO_FEATURES`` defined in ``config.py``). Adding 
``--normalize`` ensures normalization of features (recommended for ``eGeMAPS`` and ``ViT`` features).
* Options defining the model architecture: ``d_rnn``, ``rnn_n_layers``, ``rnn_bi``, ``d_fc_out``
* Options for the training process: ``--epochs``, ``--lr``, ``--seed``,  ``--n_seeds``, ``--early_stopping_patience``,
``--reduce_lr_patience``,   ``--rnn_dropout``, ``--linear_dropout``
* In order to use a GPU, please add the flag ``--use_gpu``
* Predict labels for the test set: ``--predict``
* Specific parameters for MuSe-Personalisation: ``emo_dim`` (``valence`` or ``physio-arousal``), ``win_len`` and ``hop_len`` for segmentation.

For more details, please see the ``parse_args()`` method in ``main.py``. 

The second step of the personalisation pipeline is implemented in ``personalisation.py``:
* ``--model_id``: The model to train on subject-specific data, as saved by ``main.py`` before. This ID looks like ``RNN_2023-04-11-09-11_[egemaps]_[valence]_[256_4_False_64]_[0.002_256]``, 
see the checkpoint directories created by ``main.py``
* ``--checkpoint_seed``: The specific seed of the model given by ``model_id`` (``main.py`` stores the checkpoint for every seed)
* ``emo_dim``: ``valence`` or ``physio-arousal`` 
* ``--eval_personalised``: evaluate a model previously created by ``personalisation.py``. Such models are stored in the 
same checkpoint directory as the underlying general model (as specified by ``--model_id``). This argument expects the
directory name of the personalised checkpoints, such as ``102_personalised_2023-04-11-14-36-31`` 
* The remaining arguments are analogous to those of ``main.py``

## Reproducing the baselines 
Please note that exact reproducibility can not be expected due to hardware 
### Unimodal models
For every challenge, a ``*.sh`` file is provided with the respective call (and, thus, configuration) for each of the precomputed features.
Moreover, you can directly load one of the checkpoints corresponding to the results in the baseline paper. Note that 
the checkpoints are only available to registered participants. 

A checkpoint model can be loaded and evaluated as follows:

`` main.py --task humor --feature facenet --eval_model /your/checkpoint/directory/facenet/model_102.pth`` 

Note that it is recommended to normalise egemaps and ViT features (``--normalize``).


### Late Fusion
We utilise a simple late fusion approach, which averages different models' predictions. 
First, predictions for development and test set have to be created using the ``--predict`` option in ``main.py`` or 
``personalisation.py``, respectively. This will create folders under the folder specified as prediction directory in ``config.py``.

Then, ``late_fusion.py`` merges these predictions:
* ``--task``: choose either `humor`, `mimic` or `personalisation` 
* ``--model_ids``: list of model IDs, whose predictions are to be merged. These predictions must first be created (``--predict`` in ``main.py`` or ``personalisation.py``)
* ``--seeds`` (only for MuSe-Humor and MuSe-Mimic): seeds for the respective model IDs. 
* ``--weights``: optional weights for every prediction file
* ``--emo_dim`` and ``--personalised``: Specific for MuSe-Personalisation 




##  Citation:

The MuSe2023 baseline paper is only available in a preliminary version as of now: []()

MuSe 2022 baseline paper:

```bibtex
@inproceedings{christ2022muse,
  title={The muse 2022 multimodal sentiment analysis challenge: humor, emotional reactions, and stress},
  author={Christ, Lukas and Amiriparian, Shahin and Baird, Alice and Tzirakis, Panagiotis and Kathan, Alexander and M{\"u}ller, Niklas and Stappen, Lukas and Me{\ss}ner, Eva-Maria and K{\"o}nig, Andreas and Cowen, Alan and others},
  booktitle={Proceedings of the 3rd International on Multimodal Sentiment Analysis Workshop and Challenge},
  pages={5--14},
  year={2022}
}

```



## Acknowledgement & Contributors : 
Thanks to all who contributed, especially:

<table>
  <tr>
    <td align="center">

<a href="https://github.com/lc0197"><img src="https://avatars.githubusercontent.com/u/44441963?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lukas</b></sub></a><br /><td align="center">

<a href="https://github.com/aliceebaird"><img src="https://avatars.githubusercontent.com/u/10690171?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alice</b></sub></a><br />
  </tr></table>
