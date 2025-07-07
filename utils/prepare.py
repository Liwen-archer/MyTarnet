import torch
import numpy as np
import random, math

from models.multitask_transformer_class import MultitaskTransformerModel


def initialize_training(prop):
    model = MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['dropout']).to(prop['device'])
    best_model = MultitaskTransformerModel(prop['task_type'], prop['device'], prop['nclasses'], prop['seq_len'], prop['batch'], prop['input_size'], prop['emb_size'], prop['nhead'], prop['nhid'], prop['nhid_tar'], prop['nhid_task'], prop['nlayers'], prop['dropout']).to(prop['device'])

    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss() if prop['task_type'] == 'classification' else torch.nn.MSELoss() # nn.L1Loss() for MAE
    optimizer = torch.optim.Adam(model.parameters(), lr = prop['lr'])
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr = prop['lr']) # get new optimiser

    return model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer



def attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights):
    # attention_weights = attention_weights.to('cpu')
    # instance_weights = torch.sum(attention_weights, axis = 1)
    res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1])))
    index = index.cpu().data.tolist()
    index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
    return np.array(index2)

    

def random_instance_masking(X, masking_ratio, ratio_highest_attention, instance_weights):
    indices = attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)
    boolean_indices = np.array([[True if i in index else False for i in range(X.shape[1])] for index in indices])
    boolean_indices_masked = np.repeat(boolean_indices[ : , : , np.newaxis], X.shape[2], axis = 2)
    boolean_indices_unmasked =  np.invert(boolean_indices_masked)
    
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = np.copy(X), np.copy(X), np.copy(X)
    X_train_tar = np.where(boolean_indices_unmasked, X, 0.0)
    y_train_tar_masked = y_train_tar_masked[boolean_indices_masked].reshape(X.shape[0], -1)
    y_train_tar_unmasked = y_train_tar_unmasked[boolean_indices_unmasked].reshape(X.shape[0], -1)
    X_train_tar, y_train_tar_masked, y_train_tar_unmasked = torch.as_tensor(X_train_tar).float(), torch.as_tensor(y_train_tar_masked).float(), torch.as_tensor(y_train_tar_unmasked).float()

    return X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked

    

def compute_tar_loss(model, device, criterion_tar, y_train_tar_masked, y_train_tar_unmasked, batched_input_tar, \
                    batched_boolean_indices_masked, batched_boolean_indices_unmasked, num_inst, start):
    model.train()
    out_tar = model(torch.as_tensor(batched_input_tar, device = device), 'reconstruction')[0]

    out_tar_masked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_masked)].reshape(out_tar.shape[0], -1), device = device)
    out_tar_unmasked = torch.as_tensor(out_tar[torch.as_tensor(batched_boolean_indices_unmasked)].reshape(out_tar.shape[0], -1), device = device)

    loss_tar_masked = criterion_tar(out_tar_masked[ : num_inst], torch.as_tensor(y_train_tar_masked[start : start + num_inst], device = device))
    loss_tar_unmasked = criterion_tar(out_tar_unmasked[ : num_inst], torch.as_tensor(y_train_tar_unmasked[start : start + num_inst], device = device))
    
    return loss_tar_masked, loss_tar_unmasked



def compute_task_loss(nclasses, model, device, criterion_task, y_train_task, batched_input_task, task_type, num_inst, start):
    model.train()
    out_task, attn = model(torch.as_tensor(batched_input_task, device = device), task_type)
    out_task = out_task.view(-1, nclasses) if task_type == 'classification' else out_task.squeeze()
    loss_task = criterion_task(out_task[ : num_inst], torch.as_tensor(y_train_task[start : start + num_inst], device = device)) # dtype = torch.long
    return attn, loss_task
