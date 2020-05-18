import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.helpers import get_fixed_temperature, write_tokens
from utils.dataloader import GenDataIter
from utils.preprocess import *
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.loss_functions import rsgan
from metrics.bleu import BLEU
from metrics.clas_acc import ACC
from metrics.nll import NLL

import configuration as cfg
import wandb


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
        self.clas_criterion = nn.CrossEntropyLoss()
        # Optimizer
        self.clas_opt = None
        # Metrics
        self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=True)
        self.nll_gen = NLL('NLL_gen', if_use=True, gpu=cfg.if_cuda)
        self.nll_div = NLL('NLL_div', if_use=True, gpu=cfg.if_cuda)
        self.self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=True)
        self.clas_acc = ACC(if_use=True)
        self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu]

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if cfg.GEN_PRETRAIN:
            self.gen.load_state_dict(torch.load(cfg.GEN_PR_PATH))
        self.gen = self.gen.cuda(cfg.device)
        self.dis = self.dis.to(cfg.device)

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.if_cuda:
                inp, target = inp.cuda(), target.cuda()
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
            inp, target = data['input'], data['target']
            if cfg.if_cuda:
                inp, target = inp.cuda(), target.cuda()

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
                inp, target = data['input'], data['target']
                if cfg.if_cuda:
                    inp, target = inp.cuda(), target.cuda()
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
            # self.ppl.reset(gen_tokens)
        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]



class RelGANInstructor(BasicInstructor):
    def __init__(self):
        super(RelGANInstructor, self).__init__()
        # generator, discriminator
        self.gen = RelGAN_G(mem_slots=1, num_heads=2, head_size=256, embedding_dim=32, hidden_dim=32,
                            vocab_size=len(self.word2idx_dict), max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX, gpu=cfg.if_cuda)
        self.dis = RelGAN_D(embed_dim=64, max_seq_len=cfg.MAX_SEQ_LEN, num_rep=64, vocab_size=len(self.word2idx_dict), padding_idx=cfg.PAD_IDX,
                            weights=cfg.dis_weights, gpu=cfg.if_cuda, dropout=0.25)
        self.init_model()
        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=1e-2)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=1e-4)

    def _run(self):
        #===PRE-TRAINING (GENERATOR)===
        print('Starting GENERATOR MLE TRAINING')
        self.pretrain_generator(cfg.MLE_train_epoch)
        torch.save(self.gen.state_dict(), 'data/generator_mle')

        # # ===ADVERSARIAL TRAINING===
        print('Starting Adversarial Training')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator(1)  # Generator
            d_loss = self.adv_train_discriminator(5)  # Discriminator
            self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature
            progress.set_description(
                'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))
            # TEST
            metrics = self.cal_metrics(fmt_str=False)
            wandb.log({'g_loss': g_loss, 'd_loss': d_loss,
                       'BLEU_2': metrics[0][0], 'BLEU_3': metrics[0][1], 'BLEU_4': metrics[0][2], 'BLEU_5': metrics[0][3],
                       'NLL_gen': metrics[1],
                       'NLL_div': metrics[2],
                       "Self-BLEU_2": metrics[3][0], "Self-BLEU_3": metrics[3][1], "Self-BLEU_4": metrics[3][2],
                       'epoch_adversarial': adv_epoch})
            print('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, %s' % (
                        adv_epoch, g_loss, d_loss, self.cal_metrics(fmt_str=True)))
            self._save('ADV', adv_epoch)

    def _test(self):
        print('>>> Begin test...')
        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            # ===Train===
            pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
            # ===Test===
            if epoch % 20 == 0 or epoch == epochs - 1:
                pass

            metrics = self.cal_metrics(fmt_str=False)
            wandb.log({'nll loss pretrain': pre_loss,
                       'BLEU_2': metrics[0][0], 'BLEU_3': metrics[0][1], 'BLEU_4': metrics[0][2],
                       'BLEU_5': metrics[0][3],
                       'NLL_gen': metrics[1],
                       'NLL_div': metrics[2],
                       "Self-BLEU_2": metrics[3][0], "Self-BLEU_3": metrics[3][1], "Self-BLEU_4": metrics[3][2],
                       'epoch_mle': epoch})
            print('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                        epoch, pre_loss, self.cal_metrics(fmt_str=True)))
            self._save('MLE', epoch)

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE, one_hot=True)
            if cfg.if_cuda:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, len(self.word2idx_dict)).float()
            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = rsgan(d_out_real, d_out_fake)
            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()
        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE, one_hot=True)
            if cfg.if_cuda:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, len(self.word2idx_dict)).float()
            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = rsgan(d_out_real, d_out_fake)
            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()
        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.TEMPERATURE, i, N, cfg.TEMP_ADPT)

    def _save(self, phase, epoch):
        """Save model state dict and generator's samples"""
        if phase != 'ADV':
            torch.save(self.gen.state_dict(), 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        opt.step()
