from models.instructor import BasicInstructor
import configuration as cfg
import torch.optim as optim
import torch
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
        # self.gen = LSTMGenerator(embedding_dim=200, hidden_dim=128, vocab_size=len(self.word2idx_dict),
        #                          max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX, weights='uniform')
        # self.dis = CNNDiscriminator(embed_dim=5, vocab_size=len(self.word2idx_dict), filter_sizes=[2, 3],
        #                             num_filters=[100, 100], padding_idx=cfg.PAD_IDX, gpu=cfg.if_cuda, dropout=0.2)
        self.gen = RelGAN_G(mem_slots=1, num_heads=2, head_size=256, embedding_dim=32, hidden_dim=32,
                            vocab_size=len(self.word2idx_dict), max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX,
                            gpu=cfg.if_cuda)
        self.dis = LSTM_D(vocab_size=len(self.word2idx_dict), embed_dim=64, hidden_size=32, emb_pretrained=False,
                                         max_seq_len=50, weights='uniform')
        self.init_model()
        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.GEN_PRETRAIN_LR)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=1e-4)


    def _run(self):
        self.gen.load_state_dict(torch.load('pretrained_gen.pt', map_location=cfg.device))
        print('Pretrained generator loaded')
        dis = torch.load('pretrained_dis.pth', map_location=cfg.device)
        self.dis.load_state_dict(dis['model_state_dict'])
        print('Pretrained discriminator loaded')
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
            if adv_epoch % 2 == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                self._save('ADV', adv_epoch)

    def _test(self):
        print('>>> Begin test...')
        self._run()
        pass

    def pretrain_generator(self, epochs, early_stopping=5):
        """
        Max Likelihood Pre-training for the generator
        """
        prev_loss = 100500
        es_epochs = 0
        for epoch in range(epochs):
            # ===Train===
            pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
            valid_loss = self.valid_gen_epoch(self.gen, self.valid_data.loader, self.mle_criterion)
            # ===Test===
            metrics = self.cal_metrics(fmt_str=False)
            wandb.log({'nll loss pretrain': pre_loss, 'nll loss valid':valid_loss,
                       'BLEU_2': metrics[0][0], 'BLEU_3': metrics[0][1], 'BLEU_4': metrics[0][2],
                       'BLEU_5': metrics[0][3],
                       'NLL_gen': metrics[1],
                       'NLL_div': metrics[2],
                       "Self-BLEU_2": metrics[3][0], "Self-BLEU_3": metrics[3][1], "Self-BLEU_4": metrics[3][2],
                       'epoch_mle': epoch})
            print('[MLE-GEN] epoch %d : pre_loss = %.4f, valid_loss = %.4f,  %s' % (
                        epoch, pre_loss, valid_loss, self.cal_metrics(fmt_str=True)))
            if early_stopping > 0:
                if valid_loss > prev_loss:
                    es_epochs += 1
                else:
                    es_epochs = 0
                if es_epochs >= early_stopping:
                    print('Early stopping!')
                    break

            prev_loss = min(prev_loss, valid_loss)
            self._save('MLE', epoch)


    def pretrain_discriminator(self, epochs, early_stopping=5):
        prev_loss = 100500
        es_epochs = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_dis_epoch(self.dis, self.train_data.loader, self.dis_criterion, self.dis_opt)
            valid_loss, valid_accuracy = self.valid_dis_epoch(self.dis, self.valid_data.loader, self.dis_criterion)

            wandb.log({'train_loss_dis': train_loss, 'train_accuracy_dis': train_accuracy,
                       'valid_loss_dis': valid_loss, 'valid_accuracy': valid_accuracy})
            print('[DIS-PRETRAIN] epoch %d : train_loss = %4.f, valid_loss = %4.f, train_acc = %4.f, valid_acc = %4.f') % (
                epoch, train_loss, valid_loss, train_accuracy, valid_accuracy)
            if early_stopping > 0:
                if valid_loss > prev_loss:
                    es_epochs += 1
                else:
                    es_epochs = 0
                if es_epochs >= early_stopping:
                    print('Early stopping')
                    break
            prev_loss = min(prev_loss, valid_loss)
            torch.save(self.dis.state_dict(), 'dis{}_{:05d}.pt'.format('PRE', epoch))


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

    def valid_dis_epoch(self, model, data_loader, criterion):
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
            torch.save(self.dis.state_dict(), 'dis_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE)
        write_tokens_gpt(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()
