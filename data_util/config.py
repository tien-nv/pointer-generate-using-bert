import os
import torch

from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


START_ENCODING = '[CLS]'
STOP_ENCODING = '[SEP]'
PADDING_ENCODING = '[PAD]'



root_dir = os.path.expanduser("/home/vbee/tiennv/pointer_summarizer")

#train_data_path = os.path.join(root_dir, "data/train.bin")
train_data_path = os.path.join(root_dir, "dataset/chunked/train_*")
eval_data_path = os.path.join(root_dir, "dataset/chunked/val_*")
decode_data_path = os.path.join(root_dir, "dataset/chunked/test_*")
vocab_path = os.path.join(root_dir, "dataset/vocab")
log_root = os.path.join(root_dir, "log_bert")


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
model.cuda()

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# model.eval()
# model.cuda()


# Hyperparameters
hidden_dim= 256
emb_dim= 768
batch_size= 4
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size = 180000

lr=0.15  #0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = False
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12 #cái này được thêm vào chỗ probs để nó không bị ra kết quả 0 ở chỗ probs
max_iterations = 600000
max_iterations_eval = 1000

use_gpu=True

lr_coverage=0.15  #0.15

fix_bug = False

train_log = os.path.join(log_root,'train_log.txt')

eval_log = os.path.join(log_root,'eval_log.txt')

best_model_log = os.path.join(log_root,'name_best_model_nocoverage.txt')