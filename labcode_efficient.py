import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
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
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def messagePropagate(self, lats, adj):
		return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)

	def hyperPropagate(self, lats, adj):
		lat1 = Activate(tf.transpose(adj) @ lats, self.actFunc)
		lat2 = tf.transpose(FC(tf.transpose(lat1), args.hyperNum, activation=self.actFunc)) + lat1
		lat3 = tf.transpose(FC(tf.transpose(lat2), args.hyperNum, activation=self.actFunc)) + lat2
		lat4 = tf.transpose(FC(tf.transpose(lat3), args.hyperNum, activation=self.actFunc)) + lat3
		ret = Activate(adj @ lat4, self.actFunc)
		# ret = adj @ lat4
		return ret

	def edgeDropout(self, mat):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			# newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)
		return dropOneMat(mat)

	def ours(self):
		uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uhyper = NNs.defineParam('uhyper', [args.latdim, args.hyperNum], reg=True)
		ihyper = NNs.defineParam('ihyper', [args.latdim, args.hyperNum], reg=True)
		uuHyper = (uEmbed0 @ uhyper)
		iiHyper = (iEmbed0 @ ihyper)

		ulats = [uEmbed0]
		ilats = [iEmbed0]
		gnnULats = []
		gnnILats = []
		hyperULats = []
		hyperILats = []
		for i in range(args.gnn_layer):
			ulat = self.messagePropagate(ilats[-1], self.edgeDropout(self.adj))
			ilat = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpAdj))
			hyperULat = self.hyperPropagate(ulats[-1], tf.nn.dropout(uuHyper, self.keepRate))
			hyperILat = self.hyperPropagate(ilats[-1], tf.nn.dropout(iiHyper, self.keepRate))
			gnnULats.append(ulat)
			gnnILats.append(ilat)
			hyperULats.append(hyperULat)
			hyperILats.append(hyperILat)
			ulats.append(ulat + hyperULat + ulats[-1])
			ilats.append(ilat + hyperILat + ilats[-1])
		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)

		pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
		pckIlat = tf.nn.embedding_lookup(ilat, self.iids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)

		def calcSSL(hyperLat, gnnLat):
			posScore = tf.exp(tf.reduce_sum(hyperLat * gnnLat, axis=1) / args.temp)
			negScore = tf.reduce_sum(tf.exp(gnnLat @ tf.transpose(hyperLat) / args.temp), axis=1)
			uLoss = tf.reduce_sum(-tf.log(posScore / (negScore + 1e-8) + 1e-8))
			return uLoss

		sslloss = 0
		uniqUids, _ = tf.unique(self.uids)
		uniqIids, _ = tf.unique(self.iids)
		for i in range(len(hyperULats)):
			W = NNs.defineRandomNameParam([args.latdim, args.latdim])
			pckHyperULat = tf.nn.l2_normalize(tf.nn.embedding_lookup(hyperULats[i], uniqUids), axis=1) @ W#tf.nn.l2_normalize(, axis=1)
			pckGnnULat = tf.nn.l2_normalize(tf.nn.embedding_lookup(gnnULats[i], uniqUids), axis=1)#tf.nn.l2_normalize(, axis=1)
			pckhyperILat = tf.nn.l2_normalize(tf.nn.embedding_lookup(hyperILats[i], uniqIids), axis=1) @ W#tf.nn.l2_normalize(, axis=1)
			pckGNNILat = tf.nn.l2_normalize(tf.nn.embedding_lookup(gnnILats[i], uniqIids), axis=1)#tf.nn.l2_normalize(, axis=1)
			uLoss = calcSSL(pckHyperULat, pckGnnULat)
			iLoss = calcSSL(pckhyperILat, pckGNNILat)
			sslloss += uLoss + iLoss
			
		return preds, sslloss, ulat, ilat

	def tstPred(self, ulat, ilat):
		pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
		allPreds = pckUlat @ tf.transpose(ilat)
		allPreds = allPreds * (1 - self.trnPosMask) - self.trnPosMask * 1e8
		vals, locs = tf.nn.top_k(allPreds, args.shoot)
		return locs

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		idx, data, shape = transToLsts(transpose(adj), norm=True)
		self.tpAdj = tf.sparse.SparseTensor(idx, data, shape)
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.trnPosMask = tf.placeholder(name='trnPosMask', dtype=tf.float32, shape=[None, args.item])

		self.preds, sslloss, ulat, ilat = self.ours()
		self.topLocs = self.tstPred(ulat, ilat)

		sampNum = tf.shape(self.uids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])
		negPred = tf.slice(self.preds, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize() + args.ssl_reg * sslloss
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.item)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feed_dict = {}
			uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.keepRate] = args.keepRate

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def testEpoch(self):
		epochRecall, epochNdcg = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		tstNum = 0
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}

			trnPosMask = self.handler.trnMat[batIds].toarray()
			feed_dict[self.uids] = batIds
			feed_dict[self.trnPosMask] = trnPosMask
			feed_dict[self.keepRate] = 1.0
			topLocs = self.sess.run(self.topLocs, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			recall, ndcg = self.calcRes(topLocs, self.handler.tstLocs, batIds)
			epochRecall += recall
			epochNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epochRecall / num
		ret['NDCG'] = epochNdcg / num
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
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.shoot))])
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
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()