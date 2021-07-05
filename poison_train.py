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
import random



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
	default=4096, 
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


args = parser.parse_args()
cudnn.benchmark = True
# writer = SummaryWriter() # for visualization


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
		

def bce_loss_with_logits(predictions, labels):
	return F.binary_cross_entropy(torch.sigmoid(predictions), labels, reduction='sum')


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
		
		# promoted item not in gt labels in for this user
		if len((user_items==promoted_item).nonzero()) == 0:
			# log_prob_tgt_item = 0 (not correct and using log(0) would cause loss to explode)
			continue
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


def train(args, model, optimizer, train_loader, test_loader, loss_function, poison=False, promoted_item=None):
	count, best_hr, best_poison_hr, hr_with_best_poison= 0, 0, 0, 0
	for epoch in range(args.epochs):
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item, label in train_loader:
			user = user.cuda()
			item = item.cuda()
			label = label.float().cuda()

			model.zero_grad()
			prediction = model(user, item)
			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1

		if poison:
			model.eval()
			HR, NDCG, POISON_HR = evaluate.metrics(model, test_loader, args.top_k, poison_metric=True, promoted_item=promoted_item)

			elapsed_time = time.time() - start_time
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
					time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
			print("HR: {:.3f}\tNDCG: {:.3f}\tPOISON_HR: {:.3f}\t".format(np.mean(HR), np.mean(NDCG), np.mean(POISON_HR)))

			if POISON_HR > best_poison_hr:
				hr_with_best_poison, best_poison_hr, best_ndcg, best_epoch = HR, POISON_HR, NDCG, epoch
				if args.out:
					if not os.path.exists(config.model_path):
						os.mkdir(config.model_path)
					torch.save(model, 
						'{}{}.pth'.format(config.model_path, config.model))
		else:
			model.eval()
			HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

			elapsed_time = time.time() - start_time
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
					time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
			print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

			if HR > best_hr:
				best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
				if args.out:
					if not os.path.exists(config.model_path):
						os.mkdir(config.model_path)
					torch.save(model, 
						'{}{}.pth'.format(config.model_path, config.model))

	
	if poison:
		print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, POISON_HR = {:.3f}".format(
										best_epoch, hr_with_best_poison, best_ndcg, best_poison_hr))
	else:
		print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
										best_epoch, best_hr, best_ndcg))

	return model


def insert_fake_tuple(train_data, train_mat, tuple, value):
	train_mat[tuple[0], tuple[1]] = value
	train_data.append([tuple[0], tuple[1]])
	return train_data, train_mat


##############################	LOAD DATA		##########################
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

##############################  CREATE MODEL	##########################
if config.model_name == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model_ncf = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
model_ncf.cuda()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

##############################  SET PARAMETERS	##########################

KAPPA = 1
LAMBDA = 0.01
EITA = 100
DELTA = 0.3
PROMOTED_ITEM = random.randint(0, item_num) 		# select a random item to be promoted
M = 302 											# 5% of the number of users in the dataset 
N = 30


# construct test dataloader
test_dataset = data_utils.NCFData(
	test_data, item_num, train_mat, 0, False)
test_loader = data.DataLoader(test_dataset,
	batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

selection_prob_vec = torch.from_numpy(np.ones(item_num))
items_vec = torch.arange(item_num)
items_vec = items_vec.cuda()

# iterate over each one of the fake users
for i in range(N):

	print("#############################################################################")
	print(f"#######################\t\tINSERTING FAKE USER {i}\t\t#######################")
	print("#############################################################################")
	print()

	# first insert tuple (user, promoted_item, r_max=1)
	train_data, train_mat = insert_fake_tuple(train_data, train_mat, tuple=(i, PROMOTED_ITEM), value=1)

	# construct train data loader
	train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
	train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=40)

	# pretrain poison model with loss function L (bce loss)
	model_ncf = train(args, 
						model_ncf, 
						optimizer, 
						train_loader, 
						test_loader, 
						loss_function=bce_loss_with_logits)

	# poison train the model with the poison loss function
	model_ncf = train(args, 
						model_ncf, 
						optimizer, 
						train_loader, 
						test_loader, 
						loss_function=poison_loss, 
						poison=True, 
						promoted_item=PROMOTED_ITEM)



	# if all items in selection prob vec are less than one ==> reset it
	if torch.all(selection_prob_vec < 1).item():
		selection_prob_vec = torch.from_numpy(np.ones(item_num))
	
	# get fake user vector from model
	model_ncf.eval()
	user = torch.ones(item_num) * i
	user = user.cuda()
	predictions = model_ncf(user, items_vec)
	predictions = predictions * selection_prob_vec
	values, indices = torch.topk(predictions, N)
	selection_prob_vec[indices] = selection_prob_vec[indices] * DELTA
	recommends = torch.take(items_vec, indices)
	user = torch.take(items_vec, indices)
	for u, item, value in zip(user, recommends, values):
		u = u.item()
		item = item.item()
		value = value.item()
		train_data, train_mat = insert_fake_tuple(train_data, train_mat, tuple=(u, item), value=value)
	train_data, train_mat = insert_fake_tuple(train_data, train_mat, tuple=(i, PROMOTED_ITEM), value=1)

