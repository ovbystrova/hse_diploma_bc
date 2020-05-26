import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_cuda = True if torch.cuda.is_available() else False

GEN_PRETRAIN = True  # If pretrain generator
GEN_PR_PATH = ''  # Path to pretrained generator model

MLE_train_epoch = 50  # for how long to pretrain the generator
ADV_train_epoch = 100  # how many epochs to run
TEMPERATURE = 100  # initial temperature for gumbel_somftmax
TEMP_ADPT = 'exp'  # 'exp'

CLIP_NORM = 5.0
BATCH_SIZE = 64
MAX_SEQ_LEN = 50

PAD_IDX = 28997  # {bert: 28997, gpt2: 50258, nltk:0}
START_IDX = 28996  # {bert: 28996, gpt2: 50257, nltk: 1}
PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
UNK_IDX = 28998  # {bert:28998, gpt2:50259, nltk:2}
UNK_TOKEN = '<unk>'

dis_weights = 'uniform'
LOSS_TYPE = 'rsgan'
MODEL_TYPE = 'RMC'

TEST_DATA_PATH = r"data/test_ds.txt"
DATA_PATH = r"data"
TEXTS_SIZE = '20k'  # 2k, 10k, 20k

tokenizator = 'BERT'  # ['BERT', 'NLTK', 'GPT2']
