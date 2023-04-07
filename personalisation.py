import pathlib
from datetime import datetime
from typing import List, Dict

from glob import glob, escape

import pandas as pd
from dateutil import tz
import os
from argparse import ArgumentParser
from shutil import rmtree

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from config import AROUSAL, PERSONALISATION_DIMS, PERSONALISATION
from data_parser import load_personalisation_data
from dataset import MuSeDataset, custom_collate_fn
from eval import get_predictions
from main import get_eval_fn, get_loss_fn
from train import train_personalised_models
from utils import seed_worker, log_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True, help='model id')
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, (default: {AROUSAL}).')
    # TODO this is just for internal experiments, remove before publication
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--name', type=str, default=None, help='Optional name for the new "feature set". If not given,'
                                                               'name will be calculated from the aliases.')
    parser.add_argument('--checkpoint_seed', required=True, help='Checkpoints to use, e.g. '
                                                                             '101 if for model that was trained with seed 101 '
                                                                             '(cf. output in the model directory)')
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=20,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=10,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--keep_checkpoints', action='store_true', help='Set this in order *not* to delete all the '
                                                                        'personalised checkpoints')
    # TODO add eval logic
    parser.add_argument('--eval_personalised', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')

    args = parser.parse_args()
    args.timestamp = datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = f'{args.model_id}_{args.checkpoint_seed}_personalisation_{args.timestamp}'
    args.log_file_name = args.run_name
    args.paths = {'log': os.path.join(config.LOG_FOLDER, PERSONALISATION),
                  'data': os.path.join(config.DATA_FOLDER, PERSONALISATION),
                  'model': os.path.join(config.MODEL_FOLDER, PERSONALISATION,
                                        args.log_file_name if not args.eval_personalised else os.path.join(args.model_id, args.eval_personalised))}
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    args.paths.update({'features': config.PATH_TO_FEATURES[PERSONALISATION],
                       'labels': config.PATH_TO_LABELS[PERSONALISATION],
                       'partition': config.PARTITION_FILES[PERSONALISATION]})
    args.model_file = os.path.join(config.MODEL_FOLDER, PERSONALISATION, args.model_id,
                                    f'model_{args.checkpoint_seed}.pth')
    # determine feature from filename
    args.feature = args.model_id.split("_")[2][1:-1]
    return args

def get_stats(arr):
    return {'mean':np.mean(arr), 'std':np.std(arr), 'min':np.min(arr), 'max':np.max(arr)}

def eval_personalised(personalised_cps:Dict[str, str], id2data_loaders:Dict[str, Dict[str, DataLoader]], use_gpu=False,
                      fallback_model=None):
    eval_fn, _ = get_eval_fn(PERSONALISATION)

    all_dev_labels = []
    all_dev_preds = []
    all_test_labels = []
    all_test_preds = []

    subj_dev_scores = []
    subj_test_scores = []

    fb_dev_scores = [] if fallback_model else None
    fb_test_scores = [] if fallback_model else None

    subject_ids = sorted(list(personalised_cps.keys()))
    for subject_id in subject_ids:
        model_file = personalised_cps[subject_id]
        model = torch.load(model_file, config.device)
        model.eval()

        dev_labels, subj_dev_preds = get_predictions(model=model, task=PERSONALISATION,
                                                          data_loader=id2data_loaders[subject_id]['devel'],
                                                          use_gpu=use_gpu)
        subj_dev_score = eval_fn(subj_dev_preds, dev_labels)
        subj_dev_scores.append(subj_dev_score)
        test_labels, subj_test_preds = get_predictions(model=model, task=PERSONALISATION,
                                                          data_loader=id2data_loaders[subject_id]['test'],
                                                          use_gpu=use_gpu)
        subj_test_score = eval_fn(subj_test_preds, test_labels)
        subj_test_scores.append(subj_test_score)


        if fallback_model:

            _, fb_dev_preds = get_predictions(model=fallback_model, task=PERSONALISATION,
                                                          data_loader=id2data_loaders[subject_id]['devel'],
                                                          use_gpu=use_gpu)
            fb_dev_score = eval_fn(fb_dev_preds, dev_labels)
            fb_dev_scores.append(fb_dev_score)

            _, fb_test_preds = get_predictions(model=fallback_model, task=PERSONALISATION,
                                                          data_loader=id2data_loaders[subject_id]['test'],
                                                          use_gpu=use_gpu)
            fb_test_score = eval_fn(fb_test_preds, test_labels)
            fb_test_scores.append(fb_test_score)

    if not fallback_model:
        all_dev_scores = subj_dev_scores
        all_test_scores = subj_test_scores
    else:
        fallen_back =[]
        all_dev_scores = []
        all_test_scores = []
        for i in range(len(subj_dev_scores)):
            subj_better = subj_dev_scores[i] > fb_test_scores[i]
            all_dev_scores.append(subj_dev_scores[i] if subj_better else fb_dev_scores[i])
            all_test_scores.append(subj_test_scores[i] if subj_better else fb_test_scores[i])
            fallen_back.append(not subj_better)

    # all_dev_labels = np.concatenate(all_dev_labels)
    # all_dev_preds = np.concatenate(all_dev_preds)
    # all_test_labels = np.concatenate(all_test_labels)
    # all_test_preds = np.concatenate(all_test_preds)
    #
    # eval_fn, _ = get_eval_fn(PERSONALISATION)
    # val_score = eval_fn(all_dev_preds, all_dev_labels)
    # test_score = eval_fn(all_test_preds, all_test_labels)
    val_score = np.mean(all_dev_scores)
    test_score = np.mean(all_test_scores)
    val_dict = {'personalised': get_stats(subj_dev_scores), 'overall':get_stats(all_dev_scores)}
    test_dict = {'personalised': get_stats(subj_test_scores), 'overall':get_stats(all_test_scores)}
    if fallback_model:
        val_dict.update({'fallback':get_stats(fb_dev_scores)})
        test_dict.update({'fallback':get_stats(fb_test_scores)})
    overall_dict = {'devel':val_dict, 'test':test_dict,
                    'individual_devel':{i:s for i,s in zip(subject_ids, all_dev_scores)},
                    'indvidual_test':{i:s for i,s in zip(subject_ids, all_test_scores)}}
    if fallback_model:
        overall_dict.update({'fallen_back':fallen_back})
    return all_dev_preds, val_score, all_test_preds, test_score, overall_dict

def create_data_loaders(data, test_ids):
    data_loaders = []
    for subj_data in data:
        data_loader = {}
        for partition in subj_data.keys():
            set = MuSeDataset(subj_data, partition)
            batch_size = args.batch_size if partition == 'train' else 1
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)
        data_loaders.append(data_loader)
    id2data_loaders = {i: d for i, d in zip(test_ids, data_loaders)}
    return data_loaders, id2data_loaders


def eval_trained_checkpoints(paths, feature, emo_dim, normalize, win_len, hop_len, cp_dir, use_gpu):
    data, test_ids = load_personalisation_data(paths, feature, emo_dim, normalize=normalize, win_len=win_len,
                                               hop_len=hop_len, save=True,
                                               segment_train=True)
    data_loaders, id2data_loaders = create_data_loaders(data, test_ids)
    # load personalised cps
    checkpoints = sorted([cp for cp in glob(os.path.join(escape(cp_dir), 'model_*.pth')) if 'initial' not in os.path.basename(cp)])
    initial_cp = os.path.join(cp_dir, 'model_initial.pth')
    initial_model = torch.load(initial_cp, map_location=config.device)
    initial_model.eval()
    personalised_cps = {os.path.splitext(os.path.basename(cp))[0].split("_")[1]:cp for cp in checkpoints}
    return eval_personalised(personalised_cps=personalised_cps, id2data_loaders=id2data_loaders,
                                                    use_gpu=use_gpu, fallback_model=initial_model)


def personalise(model, feature, emo_dim, temp_dir, paths, normalize, win_len, hop_len, epochs, lr, use_gpu, loss_fn,
                eval_fn, eval_metric_str, early_stopping_patience, reduce_lr_patience, seeds, regularization=0.0):
    # TODO logic for when test is not available
    data, test_ids = load_personalisation_data(paths, feature, emo_dim, normalize=normalize, win_len=win_len, hop_len=hop_len, save=True,
                              segment_train=True)
    # data_loaders = []
    # for subj_data in data:
    #     data_loader = {}
    #     for partition in subj_data.keys():
    #         set = MuSeDataset(subj_data, partition)
    #         batch_size = args.batch_size if partition == 'train' else 1
    #         shuffle = True if partition == 'train' else False  # shuffle only for train partition
    #         data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=4,
    #                                                          worker_init_fn=seed_worker, collate_fn=custom_collate_fn)
    #     data_loaders.append(data_loader)
    # id2data_loaders = {i:d for i,d in zip(test_ids, data_loaders)}
    data_loaders, id2data_loaders = create_data_loaders(data, test_ids)

    # subject id to personalised model cp
    personalised_cps = train_personalised_models(model=model, temp_dir=temp_dir, data_loaders=data_loaders, subject_ids=test_ids, epochs=epochs,
                              lr=lr, use_gpu=use_gpu, loss_fn=loss_fn, eval_fn=eval_fn,
                                eval_metric_str=eval_metric_str, early_stopping_patience=early_stopping_patience,
                              reduce_lr_patience=reduce_lr_patience, regularization = regularization, seeds=seeds)
    # all_dev_labels = []
    # all_dev_preds = []
    # all_test_labels = []
    # all_test_preds = []
    # for subject_id, model_file in personalised_cps.items():
    #     model = torch.load(model_file)
    #     subj_dev_labels, subj_dev_preds = get_predictions(model=model, task=PERSONALISATION,
    #                                                       data_loader=id2data_loaders[subject_id]['devel'], use_gpu=use_gpu)
    #     all_dev_labels.append(subj_dev_labels)
    #     all_dev_preds.append(subj_dev_preds)
    #     subj_test_labels, subj_test_preds = get_predictions(model=model, task=PERSONALISATION,
    #                                                         data_loader=id2data_loaders[subject_id]['test'], use_gpu=use_gpu)
    #     all_test_labels.append(subj_test_labels)
    #     all_test_preds.append(subj_test_preds)
    # all_dev_labels = np.concatenate(all_dev_labels)
    # all_dev_preds = np.concatenate(all_dev_preds)
    # all_test_labels = np.concatenate(all_test_labels)
    # all_test_preds = np.concatenate(all_test_preds)
    #
    # eval_fn, _ = get_eval_fn(PERSONALISATION)
    # val_score = eval_fn(all_dev_preds, all_dev_labels)
    # test_score = eval_fn(all_test_preds, all_test_labels)

    _, val_score, _, test_score,_ = eval_personalised(personalised_cps=personalised_cps, id2data_loaders=id2data_loaders,
                                                    use_gpu=use_gpu)
    return val_score, test_score


def log_personalisation_results(csv_path, params, val_score, test_score, metric_name,
                exclude_keys=['result_csv', 'cache', 'save', 'save_path', 'predict', 'eval_model', 'log_file_name']):
    '''
    Logs result of a run into a csv
    :param csv_path: path to the desired csv. Appends, if csv exists, else creates it anew
    :param params: configuration of the run (parsed cli arguments)
    :param val_results: array of validation metric results
    :param test_results: array of test metric results
    :param best_idx: index of the chosen result
    :param model_files: list of saved model files
    :param metric_name: name of the used metric
    :param exclude_keys: keys in params not to consider for logging
    :return: None
    '''
    dct = {k:[v] for k,v in vars(params).items() if not k in exclude_keys}
    dct.update({f'val_{metric_name}': val_score})
    dct.update({f'test_{metric_name}':test_score})
    #dct.update({f'mean_val_{metric_name}': np.mean(np.array(val_results))})
    #dct.update({f'std_val_{metric_name}': np.std(np.array(val_results))})
    #dct.update({f'mean_test_{metric_name}': np.mean(np.array(test_results))})
    #dct.update({f'std_test_{metric_name}': np.std(np.array(test_results))})
    ##dct.update({'model_file':model_files[best_idx]})
    df = pd.DataFrame(dct)

    # make sure the directory exists
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    # write back
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df])
    df.to_csv(csv_path, index=False)


def random_init(model:torch.nn.Module):
    for param in model.parameters():
        torch.nn.init.normal(param)
    return model


if __name__ == '__main__':
    args = parse_args()
    # TODO remove this...
    model = torch.load(args.model_file, map_location=config.device)
    if args.random_init:
        model = random_init(model)

    if not args.eval_personalised:
        pers_dir = os.path.join(config.MODEL_FOLDER, PERSONALISATION, args.model_id,
                                f'{args.checkpoint_seed}_personalised_{args.timestamp}')
        os.makedirs(pers_dir)

        eval_fn, eval_metric_str = get_eval_fn(PERSONALISATION)
        loss_fn, loss_fn_str = get_loss_fn(PERSONALISATION)
        seeds = list(range(args.seed, args.seed + args.n_seeds))
        # TODO predict logic must be in personalise
        val_score, test_score = personalise(model=model, feature=args.feature, emo_dim=args.emo_dim, temp_dir=pers_dir, paths=args.paths,
                    normalize=args.normalize, win_len=args.win_len, hop_len=args.hop_len, epochs=args.epochs, lr=args.lr,
                    use_gpu=args.use_gpu, loss_fn=loss_fn,
                    eval_fn=eval_fn, eval_metric_str=eval_metric_str, early_stopping_patience=args.early_stopping_patience,
                    reduce_lr_patience=args.reduce_lr_patience, regularization=args.regularization, seeds=seeds)
        if args.result_csv:
            log_personalisation_results(args.result_csv, params=args, metric_name=eval_metric_str, val_score=val_score,
                                        test_score=test_score)

        if not args.keep_checkpoints:
            rmtree(pers_dir)

    else:
        dev_predictions, dev_score, test_predictions, test_score, dct = eval_trained_checkpoints(
            paths=args.paths, feature=args.feature, emo_dim=args.emo_dim, normalize=args.normalize,
            win_len=args.win_len, hop_len=args.hop_len, cp_dir=args.paths['model'], use_gpu=args.use_gpu)
        # TODO print score
        print(f'[Val]: {dev_score:7.4f}')
        print(f'[Test]: {test_score:7.4f}')
        print(dct)
    # TODO also return predictions above and save them if predict
    if args.predict:
        pass
