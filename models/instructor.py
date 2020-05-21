import torch
import torch.nn as nn
from utils.dataloader import GenDataIter
from utils.preprocess import load_dict, tensor_to_tokens
from metrics.bleu import BLEU
from metrics.nll import NLL
import configuration as cfg

# TODO Упростить структуру инструкторов так, чтобы дублирующего кода было по-минимуму


class BasicInstructor:
    def __init__(self):
        self.clas = None
        # load dictionary
        self.word2idx_dict, self.idx2word_dict = load_dict(cfg.DATA_PATH)
        self.train_data = GenDataIter(cfg.DATA_PATH, batch_size=cfg.BATCH_SIZE)
        self.test_data = GenDataIter(cfg.TEST_DATA_PATH, batch_size=cfg.BATCH_SIZE, if_test_data=True)
        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        # self.clas_criterion = nn.CrossEntropyLoss()
        # Optimizer
        self.clas_opt = None
        # Metrics
        self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=True)
        self.nll_gen = NLL('NLL_gen', if_use=True, gpu=cfg.if_cuda)
        self.nll_div = NLL('NLL_div', if_use=True, gpu=cfg.if_cuda)
        self.self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=True)
        # self.clas_acc = ACC(if_use=True)
        self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu]

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        # if cfg.GEN_PRETRAIN:
        #     self.gen.load_state_dict(torch.load(cfg.GEN_PR_PATH))
        self.gen = self.gen.to(cfg.device)
        self.dis = self.dis.to(cfg.device)

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.if_cuda:
                inp, target = inp.to(cfg.device), target.to(cfg.device)
            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'].to(cfg.device), data['target'].to(cfg.device)
            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss, model)

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def eval_dis(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'].to(cfg.device), data['target'].to(cfg.device)
                pred = model.forward(inp)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

    def cal_metrics(self, fmt_str=False):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(2000, 2000)
            gen_data = GenDataIter(eval_samples, batch_size=cfg.BATCH_SIZE)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(2000, 2000), self.idx2word_dict)
            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        opt.step()
