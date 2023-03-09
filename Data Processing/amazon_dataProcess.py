import numpy as np
import json
import pickle
from Utils.TimeLogger import log
from scipy.sparse import csr_matrix
import time

def transTime(date):
	timeArr = time.strptime(date, '%Y-%m-%d %H:%M:%S')
	return time.mktime(timeArr)

def mapping(infile):
	usrId = dict()
	itmId = dict()
	usrid, itmid = [0, 0]
	interaction = list()
	with open(infile, 'r') as fs:
		for line in fs:
			arr = line.strip().split(',')
			row = arr[0]
			col = arr[1]
			timeStamp = int(arr[-1])
			if row not in usrId:
				usrId[row] = usrid
				interaction.append(dict())
				usrid += 1
			if col not in itmId:
				itmId[col] = itmid
				itmid += 1
			usr = usrId[row]
			itm = itmId[col]
			interaction[usr][itm] = timeStamp
	return interaction, usrid, itmid

# def checkFunc1(cnt):
# 	return cnt >= 10
# def checkFunc2(cnt):
# 	return cnt >= 5
# def checkFunc3(cnt):
# 	return cnt >= 5

def checkFunc1(cnt):
	return cnt >= 20
def checkFunc2(cnt):
	return cnt >= 15
def checkFunc3(cnt):
	return cnt >= 10

def filter(interaction, usrnum, itmnum, ucheckFunc, icheckFunc, filterItem=True):
	# get keep set
	usrKeep = set()
	itmKeep = set()
	itmCnt = [0] * itmnum
	for usr in range(usrnum):
		data = interaction[usr]
		usrCnt = 0
		for col in data:
			itmCnt[col] += 1
			usrCnt += 1
		if ucheckFunc(usrCnt):
			usrKeep.add(usr)
	for itm in range(itmnum):
		if not filterItem or icheckFunc(itmCnt[itm]):
			itmKeep.add(itm)

	# filter data
	retint = list()
	usrid = 0
	itmid = 0
	itmId = dict()
	for row in range(usrnum):
		if row not in usrKeep:
			continue
		usr = usrid
		usrid += 1
		retint.append(dict())
		data = interaction[row]
		for col in data:
			if col not in itmKeep:
				continue
			if col not in itmId:
				itmId[col] = itmid
				itmid += 1
			itm = itmId[col]
			retint[usr][itm] = data[col]
	return retint, usrid, itmid

def split(interaction, usrnum, itmnum):
	pickNum = 10000
	# random pick
	usrPerm = np.random.permutation(usrnum)
	pickUsr = usrPerm[:pickNum]

	tstInt = [None] * usrnum
	exception = 0
	for usr in pickUsr:
		temp = list()
		data = interaction[usr]
		for itm in data:
			temp.append((itm, data[itm]))
		if len(temp) == 0:
			exception += 1
			continue
		temp.sort(key=lambda x: x[1])
		tstInt[usr] = temp[-1][0]
		interaction[usr][tstInt[usr]] = None
	print('Exception:', exception, np.sum(np.array(tstInt)!=None))
	return interaction, tstInt

def trans(interaction, usrnum, itmnum):
	r, c, d = [list(), list(), list()]
	for usr in range(usrnum):
		if interaction[usr] == None:
			continue
		data = interaction[usr]
		for col in data:
			if data[col] != None:
				r.append(usr)
				c.append(col)
				d.append(data[col])
	intMat = csr_matrix((d, (r, c)), shape=(usrnum, itmnum))
	return intMat

prefix = 'D:/Datasets/amazon-book/'
log('Start')
interaction, usrnum, itmnum = mapping(prefix + 'ratings_Books.csv')
log('Id Mapped, usr %d, itm %d' % (usrnum, itmnum))

checkFuncs = [checkFunc1, checkFunc2, checkFunc3]
for i in range(3):
	filterItem = True if i < 2 else False
	interaction, usrnum, itmnum = filter(interaction, usrnum, itmnum, checkFuncs[i], checkFuncs[i], filterItem)
	print('Filter', i, 'times:', usrnum, itmnum)
log('Sparse Samples Filtered, usr %d, itm %d' % (usrnum, itmnum))

trnInt, tstInt = split(interaction, usrnum, itmnum)
log('Datasets Splited')
trnMat = trans(trnInt, usrnum, itmnum)
log('Train Mat Done')
with open(prefix+'trn_mat', 'wb') as fs:
	pickle.dump(trnMat, fs)
with open(prefix+'tst_int', 'wb') as fs:
	pickle.dump(tstInt, fs)
log('Interaction Data Saved')