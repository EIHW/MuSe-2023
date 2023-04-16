import time
import os
from shutil import rmtree
from typing import List, Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import device, PERSONALISATION
from eval import evaluate


def train(model, train_loader, optimizer, loss_fn, use_gpu=False):

    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        preds,_ = model(features, feature_lens)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

        loss.backward()
        optimizer.step()

        report_loss += loss.item() * batch_size
        report_size += batch_size

        total_loss += report_loss
        total_size += report_size
        report_loss, report_size, start_time = 0, 0, time.time()

    train_loss = total_loss / total_size
    return train_loss


def save_model(model, model_folder, id):
    model_file_name = f'model_{id}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, reduce_lr_patience, regularization=0.0):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=reduce_lr_patience,
                                                        factor=0.5, min_lr=1e-5, verbose=True)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            best_model_file = save_model(model, model_path, identifier)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        lr_scheduler.step(1 - np.mean(val_score))

    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file


def train_personalised_models(model, temp_dir, data_loaders:List[Dict[str, DataLoader]], subject_ids, epochs, lr, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, reduce_lr_patience, seeds, regularization=0.0):
    """
    :param model initial, general model
    :param temp_dir: model to save personalised checkpoints to
    :param data_loaders: data loaders per subject
    :param subject_ids: corresponding subject IDs
    """
    if os.path.exists(temp_dir):
        rmtree(temp_dir)
    os.makedirs(temp_dir)
    # current_seed maybe not the best parameter name here
    initial_cp = save_model(model, model_folder=temp_dir, id='initial')
    model.train()
    subj_model_files = []
    for subject_id, data_loader in zip(subject_ids, data_loaders):
        model = torch.load(initial_cp, map_location=device)
        train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']
        val_loss_before, val_score_before = evaluate(PERSONALISATION, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        print()
        print(f'Personalising for {subject_id}')
        print(f'Before personalisation | [Val] | Loss: {val_loss_before:>.4f} | [{eval_metric_str}]: {val_score_before:>7.4f}')

        best_val_score = val_score_before
        # save initial model
        subj_model_file = save_model(model, temp_dir, subject_id)

        for i,seed in enumerate(seeds):
            print(f'Seed {seed}')
            model = torch.load(initial_cp, map_location=device)
            model.train()
            torch.manual_seed(seed)
            np.random.seed(seed)
            # reshuffling of training data
            train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size,
                                                       collate_fn=train_loader.collate_fn, shuffle=True)
            _, seed_val_score, seed_model_file = train_model(model=model, task=PERSONALISATION, identifier=f'{subject_id}_{i}', data_loader=data_loader,
                                                             epochs=epochs, lr=lr, model_path=temp_dir, use_gpu=use_gpu, loss_fn=loss_fn,
                                                             eval_fn=eval_fn, eval_metric_str=eval_metric_str,
                                                             early_stopping_patience=early_stopping_patience,
                                                             reduce_lr_patience=reduce_lr_patience, regularization=regularization)

            if seed_val_score > best_val_score:
                best_val_score = seed_val_score
                # restore model
                model = torch.load(seed_model_file, map_location=device)
                # save it as the subject's model
                save_model(model, temp_dir, subject_id)
            # remove the checkpoint
            if os.path.exists(seed_model_file):
                os.remove(seed_model_file)

        print(f'After personalisation {"- personalisation did not help" if best_val_score==val_score_before else ""} '
              f'| [Val] '
              f'| [{eval_metric_str}]: {best_val_score:>7.4f} (Difference {best_val_score - val_score_before:>.4f})')
        subj_model_files.append(subj_model_file)
    return {subject_id:best_model_file for subject_id,best_model_file in zip(subject_ids, subj_model_files)}