import json
import torch
from resources.Dataset import generate_batches
from utils.HelperFunctions import *
from config import config

def train_val_test_model(model, dataset, device,  optimizer, loss_func):
    train_state = get_train_state()
    for epoch_index in range(config.num_epochs):
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: batch generator, set loss and acc to 0, set train mode on

        dataset.set_split('train')
        batch_generator = generate_batches(dataset,
                                           batch_size=config.batch_size,
                                           device=device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        print('training')
        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:
            print('training train batches')
            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = model(batch_dict['x_data'])

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('val')
        batch_generator = generate_batches(dataset,
                                           batch_size=config.batch_size,
                                           device=device)
        running_loss = 0.
        running_acc = 0.
        model.eval()
        print("validating")
        for batch_index, batch_dict in enumerate(batch_generator):
            print('validating val batches')
            # compute the output
            y_pred =  model(batch_dict['x_data'])

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        # Iterate over test dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('test')
        batch_generator = generate_batches(dataset,
                                        batch_size=config.batch_size,
                                        device=device)
        running_loss = 0.
        running_acc = 0.
        model.eval()
        print('test')
        for batch_index, batch_dict in enumerate(batch_generator):
            print('testing')
            # compute the output
            y_pred =  model(batch_dict['x_data'])

            # compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['test_loss'] = running_loss
        train_state['test_acc'] = running_acc

        # now update train_state and save model
        train_state = update_train_state(model=model,
                                         train_state=train_state)

        if train_state['stop_early']:
            break
    return train_state, model

