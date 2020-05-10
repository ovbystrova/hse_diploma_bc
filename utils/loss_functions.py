# https://github.com/williamSYSU/TextGAN-PyTorch/blob/master/utils/helpers.py
import torch
import torch.nn as nn


bce_loss = nn.BCEWithLogitsLoss()


def standart(d_out_real, d_out_fake):  # the non-satuating GAN loss
    d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
    d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))
    return g_loss, d_loss


def js(d_out_real, d_out_fake):  # the vanilla GAN loss
    d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
    d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = -d_loss_fake
    return g_loss, d_loss


def kl(d_out_real, d_out_fake):  # the GAN loss implicitly minimizing KL-divergence
    d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
    d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = torch.mean(-d_out_fake)
    return g_loss, d_loss


def hinge(d_out_real, d_out_fake):  # the hinge loss
    d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
    d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
    d_loss = d_loss_real + d_loss_fake
    g_loss = -torch.mean(d_out_fake)
    return g_loss, d_loss


def total_var(d_out_real, d_out_fake):  # the total variation distance
    d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
    g_loss = torch.mean(-nn.Tanh(d_out_fake))
    return g_loss, d_loss


def rsgan(d_out_real, d_out_fake):  # relativistic standard GAN
    d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
    g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))
    return g_loss, d_loss
