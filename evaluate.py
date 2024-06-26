import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def poison_metric(model, test_loader, top_k, promoted_item):
	POISON_HR = []
	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		POISON_HR.append(hit(promoted_item, recommends))
	return np.mean(POISON_HR)


def metrics(model, test_loader, top_k, poison_metric=False, promoted_item=None):
	HR, NDCG = [], []

	if poison_metric:
		POISON_HR = []
		assert promoted_item is not None, "We have to specify a promoted item if we want to calculate poison hit rate"

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		if poison_metric:
			POISON_HR.append(hit(promoted_item, recommends))
	
	
	if poison_metric:
		return np.mean(HR), np.mean(NDCG), np.mean(POISON_HR)
	else:
		return np.mean(HR), np.mean(NDCG)
