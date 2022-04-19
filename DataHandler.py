import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log

def transpose(mat):
	coomat = coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = 'Data/yelp/'
		elif args.data == 'ml10m':
			predir = 'Data/ml10m/'
		elif args.data == 'amazon':
			predir = 'Data/amazon/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

	def LoadData(self):
		if args.percent > 1e-8:
			print('noised')
			with open(self.predir + 'noise_%.2f' % args.percent, 'rb') as fs:
				trnMat = (pickle.load(fs) != 0).astype(np.float32)
		else:
			with open(self.trnfile, 'rb') as fs:
				trnMat = (pickle.load(fs) != 0).astype(np.float32)
		# test set
		with open(self.tstfile, 'rb') as fs:
			tstMat = pickle.load(fs)
			# tstMat = (pickle.load(fs) != 0).astype(np.float32)
		tstLocs = [None] * tstMat.shape[0]
		tstUsrs = set()
		for i in range(len(tstMat.data)):
			row = tstMat.row[i]
			col = tstMat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))

		self.trnMat = trnMat
		self.tstLocs = tstLocs
		self.tstUsrs = tstUsrs
		args.user, args.item = self.trnMat.shape
		self.prepareGlobalData()

	def prepareGlobalData(self):
		adj = self.trnMat
		adj = (adj != 0).astype(np.float32)
		self.labelP = np.squeeze(np.array(np.sum(adj, axis=0)))
		tpadj = transpose(adj)
		adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
		tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
		for i in range(adj.shape[0]):
			for j in range(adj.indptr[i], adj.indptr[i+1]):
				adj.data[j] /= adjNorm[i]
		for i in range(tpadj.shape[0]):
			for j in range(tpadj.indptr[i], tpadj.indptr[i+1]):
				tpadj.data[j] /= tpadjNorm[i]
		self.adj = adj
		self.tpadj = tpadj

	def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=args.graphSampleN, preSamp=False):
		adj = self.adj
		tpadj = self.tpadj
		def makeMask(nodes, size):
			mask = np.ones(size)
			if not nodes is None:
				mask[nodes] = 0.0
			return mask

		def updateBdgt(adj, nodes):
			if nodes is None:
				return 0
			tembat = 1000
			ret = 0
			for i in range(int(np.ceil(len(nodes) / tembat))):
				st = tembat * i
				ed = min((i+1) * tembat, len(nodes))
				temNodes = nodes[st: ed]
				ret += np.sum(adj[temNodes], axis=0)
			return ret

		def sample(budget, mask, sampNum):
			score = (mask * np.reshape(np.array(budget), [-1])) ** 2
			norm = np.sum(score)
			if norm == 0:
				return np.random.choice(len(score), 1), sampNum - 1
			score = list(score / norm)
			arrScore = np.array(score)
			posNum = np.sum(np.array(score)!=0)
			if posNum < sampNum:
				pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
				# pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
				# pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
				pckNodes = pckNodes1
			else:
				pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
			return pckNodes, max(sampNum - posNum, 0)

		def constructData(usrs, itms):
			adj = self.trnMat
			pckU = adj[usrs]
			tpPckI = transpose(pckU)[itms]
			pckTpAdj = tpPckI
			pckAdj = transpose(tpPckI)
			return pckAdj, pckTpAdj, usrs, itms

		usrMask = makeMask(pckUsrs, adj.shape[0])
		itmMask = makeMask(pckItms, adj.shape[1])
		itmBdgt = updateBdgt(adj, pckUsrs)
		if pckItms is None:
			pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
			itmMask = itmMask * makeMask(pckItms, adj.shape[1])
		usrBdgt = updateBdgt(tpadj, pckItms)
		uSampRes = 0
		iSampRes = 0
		for i in range(sampDepth + 1):
			uSamp = uSampRes + (sampNum if i < sampDepth else 0)
			iSamp = iSampRes + (sampNum if i < sampDepth else 0)
			newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
			usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
			newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
			itmMask = itmMask * makeMask(newItms, adj.shape[1])
			if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
				break
			usrBdgt += updateBdgt(tpadj, newItms)
			itmBdgt += updateBdgt(adj, newUsrs)
		usrs = np.reshape(np.argwhere(usrMask==0), [-1])
		itms = np.reshape(np.argwhere(itmMask==0), [-1])
		return constructData(usrs, itms)
