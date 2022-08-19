import logging
import os
import pdb
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json

'''Logging Modules'''
log_file_path = os.path.join(os.path.dirname(__file__), 'logs/temp.log')

def get_logger(name, log_file_path=log_file_path, logging_level=logging.INFO, log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	return logger

def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	logger.info(string)

def store_results(config, max_train_acc, max_val_acc, eq_acc, min_train_loss, best_epoch):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}
	try:
		min_train_loss = min_train_loss.item()
	except:
		pass
	# try:
	# 	min_val_loss = min_val_loss.item()
	# except:
	# 	pass
	try:
		data= {'run name' : str(config.run_name)
		, 'max val acc': str(max_val_acc)
		, 'equation acc': str(eq_acc)
		, 'max train acc': str(max_train_acc)
		, 'min train loss': str(min_train_loss)
		, 'best epoch': str(best_epoch)
		, 'epochs' : config.epochs
		, 'dataset' : config.dataset
		, 'embedding': config.embedding
		, 'embedding_size' : config.embedding_size
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'cell_type' : config.cell_type
		, 'hidden_size' : config.hidden_size
		, 'depth' : config.depth
		, 'lr' : config.lr
		, 'batch_size' : config.batch_size
		, 'dropout' : config.dropout
		}
		res_data[str(config.run_name)] = data

		with open(config.result_path, 'w', encoding='utf-8') as f:
			json.dump(res_data, f, ensure_ascii= False, indent= 4)
	except:
		print("")
		#pdb.set_trace()

def store_val_results(config, acc_score, folds_scores):
	dir = os.path.join(os.path.dirname(__file__), "cv_outputs/")
	file = f"cv_results{config.dataset}.json"
	result_path = dir + file
	try:
		with open(result_path) as f:
			res_data = json.load(f)
	except:
		res_data = {}

	try:
		folds_scores += [0 for _ in range(len(folds_scores), 5)]
		data= {'run_name' : str(config.run_name)
		, '5-fold avg acc score' : str(acc_score)
		, 'Fold0 acc' : folds_scores[0]
		, 'Fold1 acc' : folds_scores[1]
		, 'Fold2 acc' : folds_scores[2]
		, 'Fold3 acc' : folds_scores[3]
		, 'Fold4 acc' : folds_scores[4]
		, 'epochs' : config.epochs
		, 'embedding': config.embedding
		, 'embedding_size' : config.embedding_size
		, 'embedding_lr': config.emb_lr
		, 'freeze_emb': config.freeze_emb
		, 'cell_type' : config.cell_type
		, 'hidden_size' : config.hidden_size
		, 'depth' : config.depth
		, 'lr' : config.lr
		, 'batch_size' : config.batch_size
		, 'dropout' : config.dropout
		}
		#res_data[str(config.run_name)] = data
		#dir = config.val_result_path[:config.val_result_path.rindex('/')]
		res_data[str(config.run_name)] = data
		os.makedirs(dir, exist_ok=True)
		with open(result_path, 'w', encoding='utf-8') as f:
			json.dump(res_data, f, ensure_ascii= False, indent= 4)
	except:

		pdb.set_trace()

def test():
	with open('./out/CV_results_/new_asdiv-a_cs.json', 'w', encoding='utf-8') as f:
		json.dump("", f, ensure_ascii=False, indent=4)
