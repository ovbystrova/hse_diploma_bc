import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_cuda = True if torch.cuda.is_available() else False

GEN_PRETRAIN = True  # If load pretrained generator
GEN_PR_PATH = ''  # Path to pretrained generator model

MLE_train_epoch = 50  # for how long to pretrain the generator
ADV_train_epoch = 100  # how many epochs to run
TEMPERATURE = 100  # initial temperature for gumbel_somftmax
TEMP_ADPT = 'exp' # 'exp'

CLIP_NORM = 5.0
BATCH_SIZE = 64
MAX_SEQ_LEN = 50

PAD_IDX = 50258 # 0
START_IDX = 50257 #1
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
UNK_IDX = 50259 # 2
UNK_TOKEN = '<unk>'

dis_weights = 'uniform'
LOSS_TYPE = 'rsgan'
MODEL_TYPE = 'RMC'

TEST_DATA_PATH = r"data/test_ds.txt"
DATA_PATH = r"data"
TEXTS_SIZE = '20k'  # 2k, 10k, 20k

tokenizator = 'GPT2'  # ['BERT', 'NLTK', 'GPT2']
