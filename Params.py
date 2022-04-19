import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=256, type=int, help='batch size')
	parser.add_argument('--sslbatch', default=4096, type=int, help='SSL batch size')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--rank', default=4, type=int, help='embedding size')
	parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--n_factors', default=4, type=int, help='Number of factors to disentangle the original embed-size representation.')
	parser.add_argument('--n_iterations', default=2, type=int, help='Number of iterations to perform the routing mechanism.')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=20, type=int, help='K of top k')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.5, type=float, help='rate for dropout')
	parser.add_argument('--slot', default=5, type=float, help='length of time slots')
	parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
	parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')
	parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
	parser.add_argument('--hyperReg', default=1e-4, type=float, help='regularizer for hyper connections')
	parser.add_argument('--temp', default=1, type=float, help='temperature in ssl loss')
	parser.add_argument('--ssl_reg', default=1e-4, type=float, help='reg weight for ssl loss')
	parser.add_argument('--percent', default=0.0, type=float, help='percent of noise for noise robust test')
	parser.add_argument('--tstNum', default=99, type=int, help='Numer of negative samples while testing, -1 for all negatives')
	return parser.parse_args()
args = parse_args()
# tianchi
# args.user = 423423
# args.item = 874328
# beibei
# args.user = 21716
# args.item = 7977
# Tmall
# args.user = 805506#147894
# args.item = 584050#99037
# amazon
# args.user = 276163
# args.item = 270761
# ML10M_implicit
# args.user = 69878
# args.item = 10677
# yelp_implicit
# args.user = 29601
# args.item = 24734

# args.user = 78578
# args.item = 77801


# args.decay_step = args.trn_num
# args.decay_step = args.item//args.batch
args.decay_step = args.trnNum//args.batch
