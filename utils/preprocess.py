import os
import torch
import nltk
from random import choices, seed
seed(23)

DATA_PATH = r"data"
TEXTS_SIZE = '2k'  # 2k, 10k, 20k
MAX_SEQ_LEN = 50
PAD_TOKEN = '<pad>'
PAD_IDX = 0
START_TOKEN = '<start>'
BATCH_SIZE = 64


def get_tokenized(file, size=None, texts_size=TEXTS_SIZE):
    """
    tokenize [file] and sample [size] random samples
    :param texts_size: str, one of [2k, 10k, 20k]
    :param file: path to the real_data.txt file
    :param size: number of sentences
    :return: list of lists of tokens
    """
    path = "{}\\real_data_{}.txt".format(file, texts_size)
    tokenized = list()
    with open(path, encoding='utf-8') as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenized.append(text)
    if size is not None:
        tokenized = choices(tokenized, k=size)
    return tokenized


def get_word_list(tokens):
    """
    makes a lif of  unique tokens in [tokens]
    :param tokens: list of list of tokens
    :return: list of unique tokens
    """
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    """
    Get word2idx_dict and idx2word_dict
    :param word_set: list of unique tokens
    :return: dict, dict
    """
    word2idx_dict = dict()
    idx2word_dict = dict()

    index = 2
    word2idx_dict['<pad>'] = '0'  # padding token
    idx2word_dict['0'] = PAD_TOKEN
    word2idx_dict['<start>'] = '1'  # start token
    idx2word_dict['1'] = START_TOKEN

    for word in word_set:
        word2idx_dict[word] = str(index)
        idx2word_dict[str(index)] = word
        index += 1
    return word2idx_dict, idx2word_dict


def load_dict(path):
    """
    load word2index and index2word dictionaries from [path]
    :return: dict, dict
    """
    iw_path = path + '\iw_dict.txt'
    wi_path = path + '\wi_dict.txt'
    if not os.path.exists(iw_path) or not os.path.exists(iw_path):
        init_dict(path)
    with open(iw_path, 'r', encoding='utf-8') as dictin:
        idx2word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r', encoding='utf-8') as dictin:
        word2idx_dict = eval(dictin.read().strip())
    return word2idx_dict, idx2word_dict


def init_dict(path):
    """
    initialise word2index and index2word dicts. saves them to [path]
    :param path: path to data foler
    :return:
    """
    tokens = get_tokenized(path)
    word_set = get_word_list(tokens)
    word2idx_dict, idx2word_dict = get_dict(word_set)

    iw_path = path+'\iw_dict.txt'
    wi_path = path+'\wi_dict.txt'
    with open(wi_path, 'w', encoding='utf-8') as dictout:
        dictout.write(str(word2idx_dict))
    with open(iw_path, 'w', encoding='utf-8') as dictout:
        dictout.write(str(idx2word_dict))
    print('total tokens: ', len(word2idx_dict))


def tokens_to_tensor(tokens, dictionary):
    """transform word tokens to Tensor"""
    # global i
    tensor = []
    for sent in tokens:
        sent_ten = []
        for i, word in enumerate(sent):
            if word == '<pad>':
                break
            sent_ten.append(int(dictionary[str(word)]))
        while i < MAX_SEQ_LEN - 1:
            sent_ten.append(0)
            i += 1
        tensor.append(sent_ten[:MAX_SEQ_LEN])
    return torch.LongTensor(tensor)


def tensor_to_tokens(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == PAD_IDX:
                break
            sent_token.append(dictionary[str(word)])
        tokens.append(sent_token)
    return tokens
