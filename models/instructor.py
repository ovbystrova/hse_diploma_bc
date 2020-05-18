import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.helpers import get_fixed_temperature
from utils.dataloader import GenDataIter
from utils.preprocess import *
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.loss_functions import rsgan
from metrics.bleu import BLEU
from metrics.clas_acc import ACC
from metrics.nll import NLL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_cuda = True if torch.cuda.is_available() else False

GEN_PRETRAIN = False
GEN_PR_PATH = ''
SAMPLES_NUM = 100 # 10000
ADV_train_epoch = 3000
TEMPERATURE = 1
TEMP_ADPT = 'exp'
CLIP_NORM = 5.0


class BasicInstructor:
    def __init__(self):
        self.clas = None
        # load dictionary
        self.word2idx_dict, self.idx2word_dict = load_dict(DATA_PATH)

        # Dataloader
        self.train_data = GenDataIter(DATA_PATH, batch_size=BATCH_SIZE)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.clas_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.clas_opt = None

        # Metrics TODO Добавить позже
        self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=True)
        self.nll_gen = NLL('NLL_gen', if_use=True, gpu=if_cuda)
        self.nll_div = NLL('NLL_div', if_use=True, gpu=if_cuda)
        self.self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=True)
        self.clas_acc = ACC(if_use=True)
        self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu]

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if GEN_PRETRAIN:
            self.gen.load_state_dict(torch.load(GEN_PR_PATH))
        if if_cuda:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if if_cuda:
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
            if if_cuda:
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
                if if_cuda:
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
            eval_samples = self.gen.sample(SAMPLES_NUM, 4 * BATCH_SIZE)
            gen_data = GenDataIter(eval_samples, batch_size=BATCH_SIZE)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.train_data.tokens) # Заменить на test_data
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
                            vocab_size=len(self.word2idx_dict), max_seq_len=MAX_SEQ_LEN, padding_idx=PAD_IDX, gpu=if_cuda)
        self.dis = RelGAN_D(embed_dim=64, max_seq_len=MAX_SEQ_LEN, num_rep=64, vocab_size=len(self.word2idx_dict), padding_idx=PAD_IDX,
                            weights='uniform', gpu=if_cuda, dropout=0.25)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=1e-2)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=1e-4)

    def _run(self):
        #===PRE-TRAINING (GENERATOR)===
        print('Starting GENERATOR MLE TRAINING')
        self.pretrain_generator(5)
        torch.save(self.gen.state_dict(), 'data/generator_mle')

        # # ===ADVERSARIAL TRAINING===
        print('Starting Adversarial Training')
        progress = tqdm(range(ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator(1)  # Generator
            d_loss = self.adv_train_discriminator(5)  # Discriminator
            self.update_temperature(adv_epoch, ADV_train_epoch)  # update temperature

            progress.set_description(
                'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))

            # TEST
            print('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, %s' % (
                        adv_epoch, g_loss, d_loss, self.cal_metrics(fmt_str=True)))
            if adv_epoch % 20 == 0:
                pass


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
            print('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                        epoch, pre_loss, self.cal_metrics(fmt_str=True)))

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(BATCH_SIZE, BATCH_SIZE, one_hot=True)
            if if_cuda:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, len(self.word2idx_dict)).float()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
#             g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)
            g_loss, _ = rsgan(d_out_real, d_out_fake)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(BATCH_SIZE, BATCH_SIZE, one_hot=True)
            if if_cuda:
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
        self.gen.temperature = get_fixed_temperature(TEMPERATURE, i, N, TEMP_ADPT)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
