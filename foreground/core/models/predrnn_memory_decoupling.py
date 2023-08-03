__author__ = 'yunbo, jkw'

import cv2
import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_memory_decoupling import SpatioTemporalLSTMCell
import torch.nn.functional as F
from core.utils.tsne import visualization
from core.utils import preprocess, metrics
import sys
sys.path.append("./RefSR")
import RefSR
import os
import numpy as np
import time



class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path

        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        ## Ref-SR
        self.SR = RefSR.SR()

    def forward(self, frames_tensor, mask_true):
        log_time = True
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        if log_time:
            print('begin')
        for t in range(self.configs.total_length - 1):
            if log_time:
                torch.cuda.synchronize()
                beigin_time = time.time()
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            if self.visual:
                delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                if self.visual:
                    delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            if log_time:
                torch.cuda.synchronize()
                end_time = time.time()
                print('{}:{}'.format(t, end_time - beigin_time))
            ## Ref-SR
            if t < self.configs.total_length - 2 and torch.sum(mask_true[:, t]) < 1:
                print('sr:{}, ref:{}'.format(t, self.configs.input_length - 1))
                LR = x_gen.permute(0, 2, 3, 1).unsqueeze(1).clamp(0., 1.).cpu().detach().numpy()
                LR = preprocess.reshape_patch_back(LR, self.configs.patch_size)
                LR = (LR * 255.).astype('uint8')
                Ref = frames[:, self.configs.input_length - 1].permute(0, 2, 3, 1).unsqueeze(1).cpu().detach().numpy()
                # if t == self.configs.input_length - 1:
                #     Ref = frames[:, self.configs.input_length - 1].permute(0, 2, 3, 1).unsqueeze(1).cpu().detach().numpy()
                # else:
                #     Ref = next_frames[-1].permute(0, 2, 3, 1).unsqueeze(1).cpu().detach().numpy()
                Ref = preprocess.reshape_patch_back(Ref, self.configs.patch_size)
                Ref = (Ref * 255.).astype('uint8')

                LR_SR = self.SR.inference(LR[0, 0], Ref[0, 0])
                LR_SR = LR_SR.clamp(0., 1.)
                LR_SR = LR_SR.permute(1, 2, 0).cpu().numpy()

                LR_SR = np.expand_dims(np.expand_dims(LR_SR, axis=0), axis=0)
                LR_SR = preprocess.reshape_patch(LR_SR, self.configs.patch_size)
                LR_SR = torch.tensor(LR_SR[0]).to(self.configs.device).permute(0, 3, 1, 2)
                next_frames.append(LR_SR)
            else:
                next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        return next_frames, loss
