import numpy as np
import os
import torch
import math

def data_loader(dataset, task_type):
    x_train_path = os.path.join(dataset, 'X_train.npy')
    x_test_path = os.path.join(dataset, 'X_test.npy')
    if not os.path.exists(x_train_path):
        x_train_path = os.path.join(dataset, 'x_train.npy')
        x_test_path = os.path.join(dataset, 'x_test.npy')
    X_train = np.load(x_train_path, allow_pickle = True).astype(np.float64)
    X_test = np.load(x_test_path, allow_pickle = True).astype(np.float64)
    if task_type == 'classification':
        y_train = np.load(os.path.join(dataset, 'y_train.npy'), allow_pickle = True)
        y_test = np.load(os.path.join(dataset, 'y_test.npy'), allow_pickle = True)
    else:
        y_train = np.load(os.path.join(dataset, 'y_train.npy'), allow_pickle = True).as_tensor(np.float64)
        y_test = np.load(os.path.join(dataset, 'y_test.npy'), allow_pickle = True).as_tensor(np.float64)
    return X_train, y_train, X_test, y_test


def make_perfect_batch(X, num_inst, num_samples):
    extension = np.zeros((num_samples - num_inst, X.shape[1], X.shape[2]))
    X = np.concatenate((X, extension), axis = 0)
    return X



def mean_standardize_fit(X):
    m1 = np.mean(X, axis = 1)
    mean = np.mean(m1, axis = 0)
    
    s1 = np.std(X, axis = 1)
    std = np.mean(s1, axis = 0)
    
    return mean, std



def mean_standardize_transform(X, mean, std):
    return (X - mean) / std



def preprocess(prop, X_train, y_train, X_test, y_test):
    mean, std = mean_standardize_fit(X_train)
    X_train, X_test = mean_standardize_transform(X_train, mean, std), mean_standardize_transform(X_test, mean, std)

    num_train_inst, num_test_inst = X_train.shape[0], X_test.shape[0]
    num_train_samples = math.ceil(num_train_inst / prop['batch']) * prop['batch']
    num_test_samples = math.ceil(num_test_inst / prop['batch']) * prop['batch']
    
    X_train = make_perfect_batch(X_train, num_train_inst, num_train_samples)
    X_test = make_perfect_batch(X_test, num_test_inst, num_test_samples)

    X_train_task = torch.as_tensor(X_train).float()
    X_test = torch.as_tensor(X_test).float()

    if prop['task_type'] == 'classification':
        y_train_task = torch.as_tensor(y_train)
        y_test = torch.as_tensor(y_test)
    else:
        y_train_task = torch.as_tensor(y_train).float()
        y_test = torch.as_tensor(y_test).float()
    
    return X_train_task, y_train_task, X_test, y_test