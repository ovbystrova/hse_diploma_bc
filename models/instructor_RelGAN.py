import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.helpers import get_fixed_temperature, write_tokens
from utils.preprocess import tensor_to_tokens
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.loss_functions import rsgan
from models.instructor import BasicInstructor
import configuration as cfg
import wandb


class RelGANInstructor(BasicInstructor):
    def __init__(self):
        super(RelGANInstructor, self).__init__()
        # generator, discriminator
        self.gen = RelGAN_G(mem_slots=1, num_heads=2, head_size=256, embedding_dim=32, hidden_dim=32,
                            vocab_size=len(self.word2idx_dict), max_seq_len=cfg.MAX_SEQ_LEN, padding_idx=cfg.PAD_IDX,
                            gpu=cfg.if_cuda)
        self.dis = RelGAN_D(embed_dim=64, max_seq_len=cfg.MAX_SEQ_LEN, num_rep=64, vocab_size=len(self.word2idx_dict),
                            padding_idx=cfg.PAD_IDX, weights=cfg.dis_weights, gpu=cfg.if_cuda, dropout=0.25)
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
            if adv_epoch % 5 == 0 or adv_epoch == cfg.ADV_train_epoch-1:
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
            if epoch % 5 == 0 or epoch == epochs-1:
                self._save('MLE', epoch)

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target'].to(cfg.device())
            gen_samples = self.gen.sample(cfg.BATCH_SIZE, cfg.BATCH_SIZE, one_hot=True).to(cfg.device())
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
