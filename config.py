import os
from pathlib import Path

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adjust your paths here.
BASE_PATH = os.path.join(Path(__file__).parent.parent, 'MuSe-2023', 'packages')

MIMIC = 'mimic'
HUMOR = 'humor'
PERSONALISATION = 'personalisation'
TASKS = [MIMIC, HUMOR, PERSONALISATION]

PATH_TO_FEATURES = {
    MIMIC: os.path.join(BASE_PATH, 'c1_muse_mimic/features'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/feature_segments'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/feature_segments')
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {HUMOR, MIMIC}

ACTIVATION_FUNCTIONS = {
    HUMOR: torch.nn.Sigmoid,
    MIMIC: torch.nn.Sigmoid,
    PERSONALISATION:torch.nn.Tanh
}

NUM_TARGETS = {
    HUMOR: 1,
    MIMIC: 3,
    PERSONALISATION: 1
}


PATH_TO_LABELS = {
    MIMIC: os.path.join(BASE_PATH, 'c1_muse_mimic'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/label_segments'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/label_segments')
}

PATH_TO_METADATA = {
    MIMIC:os.path.join(BASE_PATH, 'c1_muse_mimic'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/metadata'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

MIMIC_LABELS = ['Approval_', 'Disappointment_', 'Uncertainty_']

# personalisation labels
AROUSAL = 'physio-arousal'
VALENCE = 'valence'
PERSONALISATION_DIMS = [AROUSAL, VALENCE]

OUTPUT_PATH = os.path.join(BASE_PATH, 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')
