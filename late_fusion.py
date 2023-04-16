import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from config import TASKS, PREDICTION_FOLDER, NUM_TARGETS, MIMIC, PERSONALISATION, AROUSAL, PERSONALISATION_DIMS, HUMOR
from eval import mean_pearsons, calc_pearsons, calc_ccc
from main import get_eval_fn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=TASKS)
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, only relevant for personalisation (default: {AROUSAL}).')
    parser.add_argument('--model_ids', nargs='+', required=True, help='model ids')
    parser.add_argument('--personalised', nargs='+', required=False,
                        help=f'Personalised model IDs for {PERSONALISATION}, '
                             f'otherwise ignored. Must be the same number as --model_ids')
    parser.add_argument('--seeds', nargs='+', required=False, help=f'seeds, needed for {MIMIC} and {HUMOR}')
    parser.add_argument('--weights', nargs='+', required=False, help='Weights for models', type=float)

    args = parser.parse_args()
    assert len(set(args.model_ids)) == len(args.model_ids), "Error, duplicate model file"
    assert len(args.model_ids) >= 2, "For late fusion, please give at least 2 different models"

    if args.weights and args.task != 'mimic':
        assert len(args.weights) == len(args.model_ids)
    elif args.weights and args.task == 'mimic':
        assert len(args.weights) == len(args.model_ids) or len(args.weights) == NUM_TARGETS['mimic'] * len(
            args.model_ids)

    if args.task == PERSONALISATION:
        assert len(args.model_ids) == len(args.personalised)
        assert args.emo_dim
    else:
        assert args.seeds

    if args.seeds and len(args.seeds) == 1:
        args.seeds = [args.seeds[0]] * len(args.model_ids)
        assert len(args.model_ids) == len(args.seeds)
    if args.task == PERSONALISATION:
        args.prediction_dirs = [
            os.path.join(PREDICTION_FOLDER, PERSONALISATION, args.emo_dim, args.model_ids[i], args.personalised[i]) for
            i in range(len(args.model_ids))]
    else:
        args.prediction_dirs = [os.path.join(PREDICTION_FOLDER, args.task, args.model_ids[i], args.seeds[i]) for i in
                                range(len(args.model_ids))]
    return args


def create_humor_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('prediction_')]].values
    if weights is None:
        weights = [1.] * pred_arr.shape[1]
    for i, w in enumerate(weights):
        preds = pred_arr[:, i]
        # normalise and weight
        # preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
        preds = w * preds
        pred_arr[:, i] = preds
    fused_preds = np.sum(pred_arr, axis=1)
    labels = df['label'].values
    return fused_preds, labels


def create_pers_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('pred')]].values
    if weights is None:
        weights = [1.] * pred_arr.shape[1]
    weights = np.array(weights) / np.sum(weights)
    for i in range(weights.shape[0]):
        preds = pred_arr[:, i]
        preds = weights[i] * preds
        pred_arr[:, i] = preds
    fused_preds = np.sum(pred_arr, axis=1)
    labels = df['label_gs'].values
    return fused_preds, labels


def create_mimic_lf(df, weights=None):
    pred_arr = df[[c for c in df.columns if c.startswith('pred')]]
    if weights is None:
        weights = [1.] * pred_arr.shape[1]
    if len(weights) < pred_arr.shape[1]:
        weights = weights * NUM_TARGETS[MIMIC]
    assert len(weights) == pred_arr.shape[1]
    num_models = int(pred_arr.shape[1] / NUM_TARGETS[MIMIC])
    weighted_preds = []
    for i in range(NUM_TARGETS[MIMIC]):
        idxs = list(range(i, pred_arr.shape[1], NUM_TARGETS[MIMIC]))
        target_preds = np.array([pred_arr.iloc[:, i].values for i in idxs]).T  # (num_examples, num_models)
        target_weights = np.array([weights[idx] for idx in idxs])
        target_weights = target_weights / np.sum(target_weights)
        for j in range(num_models):
            target_preds[:, j] = target_preds[:, j] * target_weights[j]
        weighted_preds.append(np.sum(target_preds, axis=1))
    weighted_preds = np.array(weighted_preds).T
    labels = df[[c for c in df.columns if c.startswith('gs')]].values[:, :NUM_TARGETS[MIMIC]]
    return weighted_preds, labels


# more complex eval for mimic
def eval_mimic(preds, labels):
    mean_pearson = mean_pearsons(preds, labels)
    class_wise = [calc_pearsons(preds[:, i], labels[:, i]) for i in range(labels.shape[1])]
    class_wise_cccs = [calc_ccc(preds[:, i], labels[:, i]) for i in range(labels.shape[1])]
    print(f'Mean Pearson: {mean_pearson}')
    for c in range(len(class_wise)):
        print(f'Pearson for class {c}: {class_wise[c]}')
        print(f'CCC for class {c}: {class_wise_cccs[c]}')


if __name__ == '__main__':
    args = parse_args()
    for partition in ['devel', 'test']:
        dfs = [pd.read_csv(os.path.join(pred_dir, f'predictions_{partition}.csv')) for pred_dir in args.prediction_dirs]

        meta_cols = [c for c in list(dfs[0].columns) if c.startswith('meta_')]
        for meta_col in meta_cols:
            assert all(np.all(df[meta_col].values == dfs[0][meta_col].values) for df in dfs)
        meta_df = dfs[0][meta_cols].copy()

        label_cols = [c for c in list(dfs[0].columns) if c.startswith('label')]
        for label_col in label_cols:
            assert all(np.all(df[label_col].values == dfs[0][label_col].values) for df in dfs)
        label_df = dfs[0][label_cols].copy()

        prediction_dfs = []
        for i, df in enumerate(dfs):
            pred_df = df.drop(columns=meta_cols + label_cols)
            pred_df.rename(columns={c: f'{c}_{args.model_ids[i]}' for c in pred_df.columns}, inplace=True)
            prediction_dfs.append(pred_df)
        prediction_df = pd.concat(prediction_dfs, axis='columns')

        full_df = pd.concat([meta_df, prediction_df, label_df], axis='columns')

        if args.task == 'humor':
            preds, labels = create_humor_lf(full_df, weights=args.weights)
        elif args.task == 'mimic':
            preds, labels = create_mimic_lf(full_df, weights=args.weights)
            print(partition)
            eval_mimic(preds, labels)
            print()
        elif args.task == 'personalisation':
            preds, labels = create_pers_lf(full_df, weights=args.weights)

        if args.task != MIMIC:
            eval_fn, eval_str = get_eval_fn(args.task)

            result = np.round(eval_fn(preds, labels), 4)
            print(f'{partition}: {result} {eval_str}')
