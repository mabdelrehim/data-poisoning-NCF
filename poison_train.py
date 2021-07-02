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
PROMOTED_ITEM = 121


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
		

def poison_loss(args, users, fake_user, items, promoted_item, predictions, labels, _lambda, eita, kappa):
	
	total_bce_loss = F.binary_cross_entropy(torch.sigmoid(predictions), labels, reduction='sum')

	# get predictions per user
	unique_users = torch.unique(users)
	users_loss = 0
	num_unique_users = 0
	for user in unique_users:
		
		# get predictions and corresponding items for each user
		indices = (users == user).nonzero().squeeze()
		user_predictions = torch.index_select(predictions, 0, indices)
		user_items = torch.index_select(items, 0, indices)
		user_labels = torch.index_select(labels, 0, indices)
		log_min_prediction = torch.log(torch.min(user_predictions)).item()
		
		# only consider users who have not rated promoted item yet
		indices = (user_items == promoted_item).nonzero()
		# print("indeces:",indices)
		if len(indices) != 0 and user_labels[indices[0].item()] > 0:
			continue
		
		if len((user_items==promoted_item).nonzero()) == 0:
			log_prob_tgt_item = 0
		else:
			log_prob_tgt_item = torch.log(user_predictions[(user_items==promoted_item).nonzero()[0].item()])
		
		user_loss = torch.max(torch.Tensor([log_min_prediction - log_prob_tgt_item, -1*kappa])).item()
		users_loss += user_loss
		num_unique_users += 1
	
	u = users == fake_user
	indices = u.nonzero().squeeze()
	fake_user_vector = torch.index_select(predictions, 0, indices)
	fake_user_l2_norm = torch.linalg.vector_norm(fake_user_vector)
	poison_loss = torch.pow(fake_user_l2_norm, 2) + eita*users_loss
	return total_bce_loss + _lambda*poison_loss

##############################  PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
##############################  CREATE MODEL    ##########################


