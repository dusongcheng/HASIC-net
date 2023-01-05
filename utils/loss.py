#!/usr/local/bin/python

from __future__ import division
import torch
import torch.nn as nn
import numpy as np


class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        rrmse = torch.mean(error.view(-1))
        return rrmse


class LossTrainCSS(nn.Module):
    def __init__(self):
        super(LossTrainCSS, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        rrmse = torch.mean(error.view(-1))

        grad_outputs, grad_label = self.count_grad(outputs, label)
        grad_error = torch.abs(grad_outputs - grad_label)
        grad_loss = torch.mean(grad_error.view(-1))

        # sam_grad_loss = self.Loss_SAM(grad_outputs, grad_label)

        return rrmse, grad_loss

    # def Loss_SAM(self, im_fake, im_true):
    #     N = im_true.size()[0]
    #     C = im_true.size()[1]
    #     H = im_true.size()[2]
    #     W = im_true.size()[3]
    #     nom = torch.sum(torch.mul(im_true, im_fake), dim=1)
    #     denom1 = torch.sqrt(torch.sum(torch.pow(im_true, 2), dim=1))
    #     denom2 = torch.sqrt(torch.sum(torch.pow(im_fake, 2), dim=1))
    #     sam = torch.acos(torch.div(nom, torch.mul(denom1, denom2)))
    #     sam = torch.mul(torch.div(sam, torch.tensor(np.pi)), 180)
    #     sam = torch.div(torch.sum(sam), N * H * W)
    #     return sam

    def count_grad(self, outputs, label):
        grad_outputs = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
        grad_outputs = torch.cat((outputs[:, 0:1, :, :], grad_outputs), 1)
        grad_label = label[:, 1:, :, :] - label[:, :-1, :, :]
        grad_label = torch.cat((label[:, 0:1, :, :], grad_label), 1)
        return grad_outputs, grad_label


# def rrmse_loss(outputs, label):
#     """Computes the rrmse value"""
#     error = torch.abs(outputs-label)/label
#     rrmse = torch.mean(error.view(-1))
#     return rrmse
