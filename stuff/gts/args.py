import argparse
import os
from types import SimpleNamespace as sn

import yaml
import ruamel.yaml
### Add Early Stopping ###

#FREQUENTLY CHANGED PARAMS#
import sys

#dummy default
from ..global_utils.utils import get_abs_project_path

mode = 'train' #['train', 'test']
full_cv = False
embedding = 'random'
emb_model = '' #[bert-base-uncased, roberta-base, 'UWB-AIR/Czert-B-base-cased', 'bert-base-multilingual-uncased']
en = True
epochs = 20
run_name = 'default' #[asdiv-a_en_bert, asdiv-a_en_baseline, asdiv-a_en_roberta]
dataset_name = 'default'
dataset_path = 'default'
emb_name = '' #[bert-base-uncased, roberta-base, 'UWB-AIR/Czert-B-base-cased', 'bert-base-multilingual-uncased']
embedding_size = 128
hidden_size = 512
n_layers_debth = 2
lr = 2e-3
emb_lr = 2e-3
batch_size = 16

#default
save_model = True
data_folder = "data/svamp_paper_csvs/"

def load_yaml(fl):
	path = os.path.join(os.path.dirname(__file__), fl)
	with open(path, "r") as stream:
		try:
			return ruamel.yaml.load(stream)
		except:
			return {}

def configfile_parser(config_file, config):
	assert config_file is not None
	yml = load_yaml("configs/" + config_file)
	print("embedding: " + yml['embedding'])
	if yml['embedding'] == 'bert':
		yml = {**yml, **load_yaml("configs/bert_hyperparams.yaml")}
	elif yml['embedding'] == 'roberta' or yml['embedding'] == 'xlm-roberta':
		yml = {**yml, **load_yaml("configs/roberta_hyperparams.yaml")}
	else:
		yml = {**yml, **load_yaml("configs/random_hyperparams.yaml")}

	vs = vars(config)
	for k, v in yml.items():
		vs[k] = v
	config = sn(**vs)
	#new_config = parser.
	return config

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')
	parser.add_argument('-config', type=str, default=None)
	parser.add_argument('-dataset_path', type=str, default=dataset_path)
	parser.add_argument('-data_folder', type=str, default=data_folder)

	parser.add_argument('-en', type=bool, default=en)
	parser.add_argument('-mode', type=str, default=mode, choices=['train', 'test'], help='Modes: train, test')

	# Run Config
	parser.add_argument('-dataset_name', type=str, default=dataset_name)
	parser.add_argument('-run_name', type=str, default=run_name, help='run name for logs')
	parser.add_argument('-dataset', type=str, default=dataset_path, help='Dataset')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	# parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
	parser.add_argument('-trim_threshold', type=int, default=1, help='Remove words with frequency less than this from vocab')

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=save_model)
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# Model parameters
	# parser.add_argument('-cell_type', type=str, default='gru', help='RNN cell for encoder, default: gru')
	parser.add_argument('-embedding', type=str, default=embedding, choices=['bert', 'roberta', 'word2vec', 'robeczech', 'random'], help='Embeddings')
	parser.add_argument('-emb_name', type=str, default=emb_name, choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	parser.add_argument('-embedding_size', type=int, default=embedding_size, help='Embedding dimensions of inputs')
	parser.add_argument('-emb_lr', type=float, default=emb_lr, help='Larning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)
	parser.add_argument('-word2vec_bin', type=str, default='/datadrive/global_files/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')

	parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')
	parser.add_argument('-hidden_size', type=int, default=hidden_size, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=n_layers_debth, help='Number of layers in each encoder')
	parser.add_argument('-lr', type=float, default=lr, help='Learning rate')
	parser.add_argument('-batch_size', type=int, default=batch_size, help='Batch size')
	parser.add_argument('-weight_decay', type=float, default=1e-5, help='Weight Decay')
	parser.add_argument('-beam_size', type=float, default=5, help='Beam Size')
	parser.add_argument('-epochs', type=int, default=epochs, help='Maximum # of training epochs')
	parser.add_argument('-dropout', type=float, default=0.5, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-val_result_path', type=str)
	
	# parser.add_argument('-max_length', type=int, default=100, help='Specify max decode steps: Max length string to output')
	# parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	# parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	# parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	# parser.set_defaults(bidirectional=False)
	
	# parser.add_argument('-max_grad_norm', type=float, default=0.25, help='Clip gradients to this norm')
	# parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')

	# parser.add_argument('-grade_disp', dest='grade_disp', action='store_true', help='Display grade information in validation outputs')
	# parser.add_argument('-no-grade_disp', dest='grade_disp', action='store_false', help='Don\'t display grade information')
	# parser.set_defaults(grade_disp=True)
	# parser.add_argument('-type_disp', dest='type_disp', action='store_true', help='Display Type information in validation outputs')
	# parser.add_argument('-no-type_disp', dest='type_disp', action='store_false', help='Don\'t display Type information')
	# parser.set_defaults(type_disp=True)
	parser.add_argument('-nums_disp', dest='nums_disp', action='store_true', help='Display number of numbers information in validation outputs')
	parser.add_argument('-no-nums_disp', dest='nums_disp', action='store_false', help='Don\'t display number of numbers information')
	parser.set_defaults(nums_disp=True)
	parser.add_argument('-challenge_disp', dest='challenge_disp', action='store_true', help='Display information in validation outputs')
	parser.add_argument('-no-challenge_disp', dest='challenge_disp', action='store_false', help='Don\'t display information')
	parser.set_defaults(challenge_disp=False)

	parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true', help='Calculate the train accuracy')
	parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false', help='Don\'t calculate the train accuracy')
	parser.set_defaults(show_train_acc=True)

	parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	parser.set_defaults(full_cv=full_cv)

	parser.add_argument('-len_generate_nums', type=int, default=0, help='store length of generate_nums')
	parser.add_argument('-copy_nums', type=int, default=0, help='store copy_nums')
	parser.add_argument('-crossvalid_n', type=int, default=5)
	parser.add_argument('-griffits', type=bool, default=False)
	
	return parser
