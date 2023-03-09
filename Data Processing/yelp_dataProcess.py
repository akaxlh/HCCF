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
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			data = json.loads(line.strip())
			row = data['user_id']
			col = data['business_id']
			timeStamp = transTime(data['date'])
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
	return interaction, usrid, itmid, usrId, itmId

def checkFunc1(cnt):
	return cnt >= 30
def checkFunc2(cnt):
	return cnt >= 20
def checkFunc3(cnt):
	return cnt >= 15

def filter(interaction, usrnum, itmnum, ucheckFunc, icheckFunc, filterItem, usrMap, itmMap):
	newUsrMap = dict()
	newItmMap = dict()
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
		newUsrMap[usr] = usrMap[row]
		usrid += 1
		retint.append(dict())
		data = interaction[row]
		for col in data:
			if col not in itmKeep:
				continue
			if col not in itmId:
				itmId[col] = itmid
				newItmMap[itmid] = itmMap[col]
				itmid += 1
			itm = itmId[col]
			retint[usr][itm] = data[col]
	return retint, usrid, itmid, newUsrMap, newItmMap

def split(interaction, usrnum, itmnum):
	pickNum = 10000
	# random pick
	usrPerm = np.random.permutation(usrnum)
	pickUsr = usrPerm[:pickNum]
	with open('D:/Datasets/Yelp_implicit/tst_int', 'rb') as fs:
		preTstInt = pickle.load(fs)
	pickUsr = np.reshape(np.argwhere(np.array(preTstInt) != None), [-1])
	# pickUsr = []
	# for i in range(len(preTstInt)):
	# 	if preTstInt[i] is not None:
	# 		pickUsr.append(i)

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

def makeReverseMap(mapping):
	ret = dict()
	for key in mapping:
		value = mapping[key]
		ret[value] = key
	return ret

def makeCatInfo(prefix):
	itm2cat = dict()
	with open(prefix + 'bussiness', 'r', encoding='utf-8') as fs:
		for line in fs:
			data = json.loads(line.strip())
			if 'categories' not in data or data['categories'] is None:
				continue
			itmId = data['business_id']
			cats = list(map(lambda x: x.strip(), data['categories'].split(',')))
			itm2cat[itmId] = cats
	return itm2cat

def makeUsrInfo(prefix):
	usr2usr = dict()
	with open(prefix + 'user', 'r', encoding='utf-8') as fs:
		for line in fs:
			data = json.loads(line.strip())
			if 'friends' not in data or data['friends'] is None or len(data['friends']) == 0:
				continue
			usrId = data['user_id']
			friends = list(map(lambda x: x.strip(), data['friends'].split(',')))
			usr2usr[usrId] = friends
	return usr2usr


prefix = 'D:/Datasets/Yelp_implicit_withCats/'

usr2usr = makeUsrInfo(prefix)
with open(prefix + 'usr2usr', 'wb') as fs:
	pickle.dump(usr2usr, fs)
exit()

itm2cat = makeCatInfo(prefix)
with open(prefix + 'itm2cat', 'wb') as fs:
	pickle.dump(itm2cat, fs)
exit()

log('Start')
interaction, usrnum, itmnum, usrMap, itmMap = mapping(prefix + 'review')
usrMap = makeReverseMap(usrMap)
itmMap = makeReverseMap(itmMap)
log('Id Mapped, usr %d, itm %d' % (usrnum, itmnum))

checkFuncs = [checkFunc1, checkFunc2, checkFunc3]
for i in range(3):
	filterItem = True if i < 2 else False
	interaction, usrnum, itmnum, usrMap, itmMap = filter(interaction, usrnum, itmnum, checkFuncs[i], checkFuncs[i], filterItem, usrMap, itmMap)
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
with open(prefix+'id_mapping', 'wb') as fs:
	pickle.dump({'usrMap': usrMap, 'itmMap': itmMap}, fs)
log('Mapping from new ID to original ID Saved')
