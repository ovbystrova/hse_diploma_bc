from models.instructor import BasicInstructor
import configuration as cfg
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from utils.loss_functions import rsgan
from utils.helpers import get_fixed_temperature
from utils.preprocess import tensor_to_tokens
from models.LSTM_G import LSTMGenerator
from models.CNN_D import CNNDiscriminator
from models.RelGAN_G import RelGAN_G
from models.LSTM_D import LSTM_D
from utils.helpers import write_tokens_gpt


class Instructor(BasicInstructor):
    def __init__(self):
        super(Instructor, self).__init__()
        # generator, discriminator
        self.pre_gen = LSTMGenerator(embedding_dim=200, hidden_dim=128, vocab_size=len(self.word2idx_dict),
                                 max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX, weights='uniform')
        self.gen = nn.Sequential(self.pre_gen, nn.Linear(len(self.word2idx_dict), len(self.word2idx_dict)))#, #nn.ReLU(),
                      # nn.Linear(256, len(self.word2idx_dict)))

        # self.dis = CNNDiscriminator(embed_dim=5, vocab_size=len(self.word2idx_dict), filter_sizes=[2, 3],
        #                             num_filters=[100, 100], padding_idx=cfg.PAD_IDX, gpu=cfg.if_cuda, dropout=0.2)
        # self.gen = RelGAN_G(mem_slots=1, num_heads=2, head_size=256, embedding_dim=32, hidden_dim=32,
        #                     vocab_size=len(self.word2idx_dict), max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX,
        #                     gpu=cfg.if_cuda)
        self.dis = LSTM_D(vocab_size=len(self.word2idx_dict), embed_dim=64, hidden_size=32, emb_pretrained=False,
                                         max_seq_len=50, weights='uniform')
        self.init_model()
        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.GEN_PRETRAIN_LR)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.dis_opt = optim.SGD(self.dis.parameters(), lr=1e-4)


    def _run(self):
        self.pre_gen.load_state_dict(torch.load('pretrained_gen.pt', map_location=cfg.device))
        for ind, (name, param) in enumerate(self.gen.named_parameters()):
            if ind < 7:
                param.requires_grad = False
        print('Pretrained generator loaded. Generator does not require_grad.')
        dis = torch.load('pretrained_dis.pth', map_location=cfg.device)
        self.dis.load_state_dict(dis['model_state_dict'])
        # for ind, (name, param) in enumerate(self.dis.named_parameters()):
        #         param.requires_grad = False
        print('Pretrained discriminator loaded. Discriminator does not require_grad.')
        print('Starting Adversarial Training')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator(1)  # Generator
            d_loss = self.adv_train_discriminator(5)  # Discriminator
            self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature
            progress.set_description(
                'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.pre_gen.temperature))
            # TEST
            metrics = self.cal_metrics(fmt_str=False)
            accuracy_dis = self.calc_accuracy()
            wandb.log({'g_loss': g_loss, 'd_loss': d_loss, 'diss_accuracy': accuracy_dis,
                       'BLEU_2': metrics[0][0], 'BLEU_3': metrics[0][1], 'BLEU_4': metrics[0][2], 'BLEU_5': metrics[0][3],
                       'NLL_gen': metrics[1],
                       'NLL_div': metrics[2],
                       "Self-BLEU_2": metrics[3][0], "Self-BLEU_3": metrics[3][1], "Self-BLEU_4": metrics[3][2],
                       'epoch_adversarial': adv_epoch})
            # wandb.log({'diss_accuracy': accuracy_dis,
            #            'g_loss': g_loss,
            #            'd_loss': d_loss})
            if adv_epoch % 2 == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                self._save('ADV', adv_epoch)

    def _test(self):
        print('>>> Begin test...')
        self._run()
        pass

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.pre_gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE, one_hot=True)
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
            gen_samples = self.pre_gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE, one_hot=True)
            if cfg.if_cuda:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, len(self.word2idx_dict)).float()
            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = rsgan(d_out_real, d_out_fake)
            # self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()
        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.pre_gen.temperature = get_fixed_temperature(cfg.TEMPERATURE, i, N, cfg.TEMP_ADPT)

    def _save(self, phase, epoch):
        """Save model state dict and generator's samples"""
        if phase == 'ADV':
            torch.save(self.gen.state_dict(), 'gen_{}_{:05d}.pt'.format(phase, epoch))
            torch.save(self.dis.state_dict(), 'dis_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.pre_gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE)
        write_tokens_gpt(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))


    def calc_accuracy(self):
        total_acc = 0
        for i in range(10):
            sampled = self.pre_gen.sample(64, 64, one_hot=True)
            real = F.one_hot(self.test_data.random_batch()['target'], len(self.word2idx_dict)).float()
            to_dis = torch.cat((sampled, real))
            target_gen = torch.zeros((cfg.BATCH_SIZE)).to(cfg.device)
            target_dis = torch.ones((cfg.BATCH_SIZE)).to(cfg.device)
            target = torch.cat((target_gen, target_dis))
            with torch.no_grad():
                logits = self.dis(to_dis)
                total_acc += torch.sum((logits.argmax(dim=-1) == target)).item()
        return total_acc / (128*10)


    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()
