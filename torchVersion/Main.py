import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import setproctitle

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			log('Model Initialized')
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, tstFlag))
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def prepareModel(self):
		self.model = Model().cuda()
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epPreLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			bprLoss, sslLoss = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj, args.keepRate)
			sslLoss = sslLoss * args.ssl_reg

			regLoss = calcRegLoss(self.model) * args.reg
			loss = bprLoss + regLoss + sslLoss
			epLoss += loss.item()
			epPreLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['preLoss'] = epPreLoss / steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epLoss, epRecall, epNdcg = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = t.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		recallBig = 0
		ndcgBig =0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('../History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		content = {
			'model': self.model,
		}
		t.save(content, '../Models/' + args.save_path + '.mod')
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		ckp = t.load('../Models/' + args.load_model + '.mod')
		self.model = ckp['model']
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		with open('../History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	setproctitle.setproctitle('akaxia')
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()