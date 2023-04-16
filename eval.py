import audmetric
import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from scipy import stats

from config import MIMIC_LABELS, MIMIC, PERSONALISATION

def calc_ccc(preds, labels):
    return audmetric.concordance_cc(preds, labels)


def calc_pearsons(preds, labels):
    r = stats.pearsonr(preds, labels)
    return r[0]


def mean_pearsons(preds, labels):
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_r = np.array([calc_pearsons(preds[:, i], labels[:, i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r


def calc_auc(preds, labels):
    if not type(preds) == np.ndarray:
        preds = np.concatenate(preds)
    if not type(labels) == np.ndarray:
        labels = np.concatenate(labels)

    fpr, tpr, thresholds = roc_curve(labels, preds)
    return auc(fpr, tpr)


def write_mimic_predictions(full_metas, full_preds, full_labels, csv_dir, filename):
    meta_arr = np.row_stack(full_metas).squeeze()
    preds_arr = np.row_stack(full_preds)
    labels_arr = np.row_stack(full_labels)
    pred_cols = [f'pred_{l}' for l in MIMIC_LABELS]
    label_cols = [f'gs_{l}' for l in MIMIC_LABELS]
    pred_df = pd.DataFrame(columns=['File_ID'] + pred_cols + label_cols)
    pred_df['File_ID'] = meta_arr
    pred_df[pred_cols] = preds_arr
    pred_df[label_cols] = labels_arr

    pred_df.to_csv(os.path.join(csv_dir, filename), index=False)
    return None


def write_predictions(task, full_metas, full_preds, full_labels, prediction_path, filename):
    assert prediction_path != ''

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if task == MIMIC:
        # TODO adapt
        return write_mimic_predictions(full_metas, full_preds, full_labels, prediction_path, filename)

    metas_flat = []
    for meta in full_metas:
        metas_flat.extend(meta)
    preds_flat = []
    for pred in full_preds:
        preds_flat.extend(pred if isinstance(pred, list) else (pred.squeeze() if pred.ndim > 1 else pred))

    labels_flat = []
    for label in full_labels:
        labels_flat.extend(label if isinstance(label, list) else (label.squeeze() if label.ndim > 1 else label))

    if isinstance(metas_flat[0], list):
        num_meta_cols = len(metas_flat[0])
    else:
        # np array
        num_meta_cols = metas_flat[0].shape[0]
    prediction_df = pd.DataFrame(columns=[f'meta_col_{i}' for i in range(num_meta_cols)])
    for i in range(num_meta_cols):
        prediction_df[f'meta_col_{i}'] = [m[i] for m in metas_flat]
    prediction_df['prediction'] = preds_flat
    prediction_df['label'] = labels_flat
    prediction_df.to_csv(os.path.join(prediction_path, filename), index=False)


def get_predictions(model, task, data_loader, use_gpu=False):
    full_preds = []
    full_labels = []
    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data

            batch_size = features.size(0) if task != PERSONALISATION else 1

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()

            preds,_ = model(features, feature_lens)

            # only relevant for stress
            feature_lens = feature_lens.detach().cpu().tolist()
            cutoff = feature_lens[0] if task == PERSONALISATION else batch_size

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist()[:cutoff])
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist()[:cutoff])
    if task == PERSONALISATION:
        full_preds = flatten_personalisation_for_ccc(full_preds)
        full_labels = flatten_personalisation_for_ccc(full_labels)
    return full_labels, full_preds


def evaluate(task, model, data_loader, loss_fn, eval_fn, use_gpu=False, predict=False, prediction_path=None,
             filename=None):
    losses, sizes = 0, 0
    full_preds = []
    full_labels = []
    if predict:
        full_metas = []
    else:
        full_metas = None

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            if not predict:
                if torch.any(torch.isnan(labels)):
                    print('No labels available, no evaluation')
                    return np.nan, np.nan

            batch_size = features.size(0) if task != PERSONALISATION else 1

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()

            preds,_ = model(features, feature_lens)

            # only relevant for personalisation
            feature_lens = feature_lens.detach().cpu().tolist()
            cutoff = feature_lens[0] if task == PERSONALISATION else batch_size
            if predict:
                full_metas.append(metas.tolist()[:cutoff])

            loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

            losses += loss.item() * batch_size
            sizes += batch_size

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist()[:cutoff])
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist()[:cutoff])

        if predict:
            write_predictions(task, full_metas, full_preds, full_labels, prediction_path, filename)
            return
        else:
            if task == PERSONALISATION:
                full_preds = flatten_personalisation_for_ccc(full_preds)
                full_labels = flatten_personalisation_for_ccc(full_labels)
            score = eval_fn(full_preds, full_labels)
            total_loss = losses / sizes
            return total_loss, score


def flatten_personalisation_for_ccc(lst):
    """
    Brings full_preds and full_labels of stress into the right format for the CCC function
    :param lst: list of lists of different lengths
    :return: flattened numpy array
    """
    return np.concatenate([np.array(l) for l in lst])