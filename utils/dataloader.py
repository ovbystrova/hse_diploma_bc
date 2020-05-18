import random
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_cuda = True if torch.cuda.is_available() else False

class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GenDataIter:
    def __init__(self, data,
                 batch_size,
                 shuffle=None):
        self.batch_size = batch_size
        self.start_letter = '<start>'
        self.shuffle = shuffle
        self.word2idx, self.idx2word = load_dict(DATA_PATH)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(data)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        elif isinstance(samples, str):  # filename
            inp, target = self.load_data(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        # inp, target = self.load_data(samples)
        # all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=if_cuda):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = 1
        inp[:, 1:] = target[:, :MAX_SEQ_LEN - 1]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenized(filename)
        samples_index = tokens_to_tensor(self.tokens, self.word2idx)
        # tokens = get_tokenized(filename)
        # samples_index = tokens_to_tensor(tokens, self.word2idx)
        return self.prepare(samples_index)
