from tokenizers import SentencePieceBPETokenizer
import re

PATH = 'wikitext-2'


def open_file(name):
    with open(name, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def save_file(text, name):
    with open('{}.txt'.format(name), 'w', encoding='utf-8') as file:
        file.write(text)


def clean_data(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'&[a-z]{0,7};', ' ', text)
    text = re.sub(r'\s{2,10}', ' ', text)
    text = re.sub(r'\s{2,10}', ' ', text)
    text = re.sub(r'\\\\x\d{1,4}', '', text)
    text = re.sub(r'=\s.{3,15}=\s', '', text)
    text = re.sub(r'\s@', ' ', text)
    text = re.sub(r'@\s', ' ', text)
    return text


def special_tokens(text):
    """
    Adds <eos> token if eos symbol occures.
    """
    eos = ['.', ' !', ' ?', ' ? !', ' \'n', ' .']
    eos_token = '<eos> <start>'
    for element in eos:
        text = text.replace(element, ' ' + eos_token + ' ')
    text = re.sub(r'<start>\s{2,10}<eos>', ' ', text)
    while '  ' in text:
        text = re.sub(r'\s{2,10}', ' ', text)
    text = '<start> ' + text
    return text


def make_tokenizer(path=PATH):
    """
    Cleans data and adds special tokens for snt_piece training
    Trains SentencePieceBPETokenizer
    :param path: relative path to wikitext-2 folder
    :return: SentencePieceBPETokenizer
    """
    try:
        wiki_train = open_file('{}/wiki.train.tokens'.format(path))
    except:
        import os
        os.chdir('..')
        wiki_train = open_file('{}/wiki.train.tokens'.format(path))
    wiki_train = clean_data(wiki_train)
    wiki_train = special_tokens(wiki_train)
    save_file(wiki_train, '{}/wiki_train'.format(path))

    wiki_valid = open_file('{}/wiki.valid.tokens'.format(path))
    wiki_valid = clean_data(wiki_valid)
    wiki_valid = special_tokens(wiki_valid)
    save_file(wiki_valid, '{}/wiki_valid'.format(path))

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(['{}/wiki_train.txt'.format(path), '{}/wiki_valid.txt'.format(path)],
                    special_tokens=['<eos>', '<unk>', '<start>'], vocab_size=30000)
    return tokenizer


def tokenize(text, tokenizer):
    text = clean_data(text)
    text = special_tokens(text)
    return tokenizer.encode(text).tokens


if __name__ == '__main__':
    PATH = 'wikitext-2'
    tokenizer = make_tokenizer(PATH)

    text = 'One two three four five vyshel zaika pogulat. And someone ate him'
    # text = special_tokens(clean_data(text))

    print(tokenize(text, tokenizer))
