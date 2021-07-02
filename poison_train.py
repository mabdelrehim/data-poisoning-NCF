import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F

import model
import config
import evaluate
import data_utils



parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epochs")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--fake_users", 
	type=int,
	default=302,  
	help="number of inserted fake users (default is 302 which is 5% of the total number of users)")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
# writer = SummaryWriter() # for visualization

KAPPA = 1
LAMBDA = 0.01
EITA = 100
DELTA = 0.3


def prepare_data(args, train_data, test_data, train_mat, item_num):
    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
    train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=40)
    test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

    return train_loader, test_loader

def create_model(args):
    if config.model_name == 'NeuMF-pre':
	    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	    GMF_model = torch.load(config.GMF_model_path)
	    MLP_model = torch.load(config.MLP_model_path)
    else:
	    GMF_model = None
	    MLP_model = None

    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
    model.cuda()

    if config.model == 'NeuMF-pre':
	    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
	    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def bce_loss(args, predictions, labels):
    total_bce_loss = torch.sum(-labels * torch.log(predictions) - (1 - labels) * torch.log(1 - predictions))
    # Getting the mean BCE loss
    num_of_samples = predictions.shape[0]
    mean_bce_loss = total_bce_loss / num_of_samples
    return mean_bce_loss

def poison_loss(args, users, items, predictions, labels, lambda, eita):
	total_bce_loss = torch.sum(-labels * torch.log(predictions) - (1 - labels) * torch.log(1 - predictions))
    # Getting the mean BCE loss
	num_of_samples = predictions.shape[0]
	mean_bce_loss = total_bce_loss / num_of_samples

	# get predictions per user
	unique_users = torch.unique(users)
	user_loss = 0
	for user in unique_users:
		b = users == user
		indices = b.nonzero()
		user_predictions = torch.index_select(predictions, 0, indices)
		user_items = torch.index_select(items, 0, indices)
	
	return mean_bce_loss + lambda*mean_poison_loss

##############################  PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
##############################  CREATE MODEL    ##########################


