import os
import torch
import nltk
import configuration as cfg
from random import choices, seed
from transformers import GPT2Tokenizer, DistilBertTokenizer
seed(23)


def get_tokenized(file, size=None, texts_size=cfg.TEXTS_SIZE, if_test=False, if_valid=False):
    """
    tokenize [file] and sample [size] random samples
    :param if_valid: True if creating Validation loader
    :param if_test: True if creating Tets loader
    :param texts_size: str, one of [2k, 10k, 20k]
    :param file: path to the real_data.txt file
    :param size: number of sentences
    :return: list of lists of tokens
    """
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', unk_token='<unk>', eos_token='<pad>',
                                              pad_token='<pad>', bos_token='<start>')
    tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-cased', unk_token='<unk>', eos_token='<pad>',
                                              pad_token='<pad>', bos_token='<start>')
    if if_test or if_valid:
        path = file
    else:
        path = "{}/real_data_{}.txt".format(file, texts_size)
    tokenized = list()
    with open(path, encoding='utf-8') as raw:
        for text in raw:
            if cfg.tokenizator == 'GPT2':
                text = tokenizer_gpt2.tokenize(text)
            elif cfg.tokenizator == 'BERT':
                text = tokenizer_bert.tokenize(text)
            else:  # nltk tokenizer as default choice
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

    index = 3
    word2idx_dict['<pad>'] = '0'
    idx2word_dict['0'] = cfg.PAD_TOKEN
    word2idx_dict['<start>'] = '1'
    idx2word_dict['1'] = cfg.START_TOKEN
    word2idx_dict['<unk>'] = '2'
    idx2word_dict['2'] = cfg.UNK_TOKEN
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
    iw_path = path + '/iw_{}.txt'.format(cfg.tokenizator)
    wi_path = path + '/wi_{}.txt'.format(cfg.tokenizator)
    if not os.path.exists(iw_path) or not os.path.exists(iw_path):
        init_dict(path)
    with open(iw_path, 'r', encoding='utf-8') as dictin:
        idx2word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r', encoding='utf-8') as dictin:
        word2idx_dict = eval(dictin.read().strip())
    return word2idx_dict, idx2word_dict


def load_test_dict():
    """Build test data dictionary, extend from train data. For the classifier."""
    word2idx_dict, idx2word_dict = load_dict(cfg.DATA_PATH)  # train dict
    tokens = get_tokenized(cfg.TEST_DATA_PATH, if_test=True)
    word_set = get_word_list(tokens)
    index = len(word2idx_dict)
    # extend dict with test data
    for word in word_set:
        if word not in word2idx_dict:
            word2idx_dict[word] = str(index)
            idx2word_dict[str(index)] = word
            index += 1
    return word2idx_dict, idx2word_dict


def init_dict(path):
    """
    initialise word2index and index2word dicts. saves them to [path]
    :param path: path to data folder
    :return:
    """
    tokens = get_tokenized(path)
    word_set = get_word_list(tokens)
    word2idx_dict, idx2word_dict = get_dict(word_set)
    iw_path = path + '/iw_{}_{}.txt'.format(cfg.tokenizator, cfg.TEXTS_SIZE)
    wi_path = path + '/wi_{}_{}.txt'.format(cfg.tokenizator, cfg.TEXTS_SIZE)
    with open(wi_path, 'w', encoding='utf-8') as dictout:
        dictout.write(str(word2idx_dict))
    with open(iw_path, 'w', encoding='utf-8') as dictout:
        dictout.write(str(idx2word_dict))
    print('total tokens: ', len(word2idx_dict))


def tokens_to_tensor(tokens, dictionary):
    """transform word tokens to Tensor"""
    tensor = []
    for sent in tokens:
        sent_ten = []
        for i, word in enumerate(sent):
            if word == cfg.PAD_TOKEN:
                break
            try:
                sent_ten.append(int(dictionary[str(word)]))
            except:
                sent_ten.append(cfg.UNK_IDX)
        while i < cfg.MAX_SEQ_LEN - 1:
            sent_ten.append(cfg.PAD_IDX)
            i += 1
        tensor.append(sent_ten[:cfg.MAX_SEQ_LEN])
    return torch.LongTensor(tensor)


def tensor_to_tokens(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == cfg.PAD_IDX:
                break
            sent_token.append(dictionary[str(int(word))])
        tokens.append(sent_token)
    return tokens
