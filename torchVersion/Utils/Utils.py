import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return -t.log(nume / deno).mean()