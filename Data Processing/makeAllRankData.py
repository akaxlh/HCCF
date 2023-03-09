from Params import *
from DataHandler import DataHandler
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import pickle

handler = DataHandler()
handler.LoadData()

trnMat = coo_matrix(handler.trnMat)
tstInt = handler.tstInt
row = list(trnMat.row)
col = list(trnMat.col)
data = list(trnMat.data)
for i in range(args.user):
	if tstInt[i] is not None:
		row.append(i)
		col.append(tstInt[i])
		data.append(1)

row = np.array(row)
col = np.array(col)
data = np.array(data)

leng = len(row)
indices = np.random.permutation(leng)
trn = int(leng * 0.7)
val = int(leng * 0.8)

trnIndices = indices[:trn]
trnMat = coo_matrix((data[trnIndices], (row[trnIndices], col[trnIndices])), shape=[args.user, args.item])

valIndices = indices[trn:val]
valMat = coo_matrix((data[valIndices], (row[valIndices], col[valIndices])), shape=[args.user, args.item])

tstIndices = indices[val:]
tstMat = coo_matrix((data[tstIndices], (row[tstIndices], col[tstIndices])), shape=[args.user, args.item])

with open('data/%s/trnMat.pkl' % args.data, 'wb') as fs:
	pickle.dump(trnMat, fs)
with open('data/%s/valMat.pkl' % args.data, 'wb') as fs:
	pickle.dump(valMat, fs)
with open('data/%s/tstMat.pkl' % args.data, 'wb') as fs:
	pickle.dump(tstMat, fs)