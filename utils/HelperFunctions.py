import json
import numpy as np
import torch
from config import config

def get_train_state():
    return  {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'epoch_index': 0,
            'best_epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}

def update_train_state(model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), config.save_model_directory+config.model_filename+str(train_state['best_epoch_index']))
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # update stopping variables
            train_state['best_epoch_index'] = train_state['epoch_index']
            train_state['early_stopping_best_val'] = loss_t
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), config.save_model_directory+config.model_filename+str(train_state['best_epoch_index']))

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ? if early_stopping step >= early_stopping_criteria, this is True and will stop loop
        train_state['stop_early'] = train_state['early_stopping_step'] >= config.early_stopping_criteria
        with open('results/training_results.json', 'w') as fp:
            json.dump(train_state, fp)

    return train_state

def save_embeddings(model, train_state, words):
    """save the word embeddings from the model

    Args:
        train_state (dictionary): dictionary representing the training state values
    """
    # load best model and the embeddings
    model.load_state_dict(torch.load(config.save_model_directory+config.model_filename+str(train_state['best_epoch_index'])))
    embeddings = model.embedding.weight.data
    # save embeddings as .txt, just like GloVe embeddings
    with open(config.save_embeddings_directory + config.embeddings_filename +str(embeddings.shape[1])+"d_"+str(train_state['best_epoch_index']) + 'epoch.txt', 'w') as f:
        for i in range(embeddings.shape[0]):
            f.write("%s" % words[i])
            for j in range(embeddings.shape[1]):
                f.write(" %s" % embeddings.numpy()[i][j])
            f.write("\n")

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


