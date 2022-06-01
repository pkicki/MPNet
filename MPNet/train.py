import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_dataset
from model import MLP
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import math



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_input(i, data, targets, bs):
    if i + bs < len(data):
        bi = data[i:i + bs]
        bt = targets[i:i + bs]
    else:
        bi = data[i:]
        bt = targets[i:]

    return torch.from_numpy(bi), torch.from_numpy(bt)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    bs = 64
    num_epochs = 10000
    save_step = 100
    #input_size = 14
    #output_size = 7
    input_size = 28
    output_size = 14
    stride = 5
    #stride = 1
    model_name = f'mlp_bs{bs}_stride{stride}_PReLU'
    ds_path = f"../datasets/paper/train/"
    ds_val_path = ds_path.replace("train", "val")
    # Build data loader
    dataset, targets = load_dataset(ds_path)
    dataset_val, targets_val = load_dataset(ds_val_path)

    # Build the models
    mlp = MLP(input_size, output_size)

    if torch.cuda.is_available():
        mlp.cuda()

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(mlp.parameters())

    writer = SummaryWriter()
    # Train the Models
    total_loss = []
    print(len(dataset))
    print(len(targets))
    sm = save_step # start saving models after 100 epochs
    best_val_loss = 1e10
    best_idx = 0
    for epoch in range(1, num_epochs+1):
        print("epoch" + str(epoch))
        # train
        avg_loss = 0
        for i in range(0, len(dataset), bs):
            # Forward, Backward and Optimize
            mlp.zero_grad()
            bi, bt = get_input(i, dataset, targets, bs)
            bi = to_var(bi)
            bt = to_var(bt)
            bo = mlp(bi)
            loss = criterion(bo, bt)
            #avg_loss = avg_loss + loss.data[0]
            avg_loss = avg_loss + loss.data.detach().cpu().numpy()
            loss.backward()
            optimizer.step()
        print("--average loss:")
        epoch_loss = avg_loss / (len(dataset) / bs)
        print(epoch_loss)
        total_loss.append(epoch_loss)
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        # val
        avg_loss = 0
        for i in range(0, len(dataset_val), bs):
            # Forward, Backward and Optimize
            bi, bt = get_input(i, dataset_val, targets_val, bs)
            bi = to_var(bi)
            bt = to_var(bt)
            bo = mlp(bi)
            loss = criterion(bo, bt)
            avg_loss = avg_loss + loss.data.detach().cpu().numpy()
        print("--average validation loss:")
        epoch_loss = avg_loss / (len(dataset_val) / bs)
        print(epoch_loss)
        writer.add_scalar('Loss/epoch_val', epoch_loss, epoch)
        # Save the models
        if epoch == sm:
            model_path = model_name + "_no_" + str(sm) + '.pkl'
            torch.save(mlp.state_dict(), os.path.join(args.model_path, model_path))
            sm = sm + save_step  # save model after every 50 epochs from 100 epoch ownwards
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            model_path = model_name + "_best_" + str(best_idx) + '.pkl'
            torch.save(mlp.state_dict(), os.path.join(args.model_path, model_path))
            best_idx += 1

    torch.save(total_loss, 'total_loss.dat')
    model_path = model_name + '_final.pkl'
    torch.save(mlp.state_dict(), os.path.join(args.model_path, model_path))


if __name__ == '__main__':
    print("XD")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--ds_path', type=str, default='../datasets/dummy', help='directory with data')
    parser.add_argument('--no_env', type=int, default=50, help='directory for obstacle images')
    parser.add_argument('--no_motion_paths', type=int, default=2000, help='number of optimal paths in each environment')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=50, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--input_size', type=int, default=28, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int, default=14, help='dimension of the output vector')
    parser.add_argument('--hidden_size', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()
    print(args)
    main(args)
