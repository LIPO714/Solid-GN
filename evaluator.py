
import copy
import datetime
import random
import time

import imageio
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
from config import _C as C
from utils import tprint, pprint, sort_for_poss, flip_horizontally
import os
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import utils


class PredEvaluator(object):
    def __init__(self, device, data_loader, model, output_dir, exp_name, output_fig=False, if_sort=False, checkpoint=None,
                 loss_max_num=100, dataset="Crash", pred_all=True):
        self.device = device
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.model = model
        self.if_sort = if_sort
        self.loss_max_num = loss_max_num
        self.output_fig = output_fig
        self.dataset = dataset
        self.pred_all = pred_all
        self.exp_name = exp_name

        self.metadata = utils.update_metadata(self.data_loader.dataset.metadata, self.device)

    def test(self):
        torch.set_printoptions(precision=16)
        self.model.eval()

        loss_num = 0
        loss_sum = 0
        acc_loss_sum = 0
        flip_loss_sum = 0

        start_dt = time.time()
        print("start_datetime:", start_dt)

        for batch_idx, data in enumerate(self.data_loader):
            assert data[0].shape[0] == 1

            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            poss, vels, tgt_accs, tgt_vels, particle_r, particle_type, nonk_mask, tgt_poss, idx, particle_m = data

            tprint(f'eval: {batch_idx}/{len(self.data_loader)} ')

            with torch.no_grad():
                num_rollouts = tgt_poss.shape[1]
                poss_flip, vels_flip, particle_type_flip, nonk_mask_flip, tgt_poss_flip, tgt_vels_flip = self.flip(
                    poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels, num_rollouts=num_rollouts)

                outputs = self.model(poss, vels, particle_r, particle_type, nonk_mask, self.metadata, tgt_poss,
                                     tgt_vels, num_rollouts=num_rollouts)

                if outputs['pred_collaposed']:
                    print('Rollout collaposed!!!!!!!!!!!!!')
                    continue

                tgt_accns = (tgt_accs - self.metadata['acc_mean']) / self.metadata['acc_std']

                labels = {
                    'accns': tgt_accns,
                    'poss': tgt_poss,
                    'vels': tgt_vels
                }
                # 计算loss
                loss, loss_steps, acc_loss, acc_loss_steps = self.loss(outputs, labels, nonk_mask)

                loss_num += 1
                loss_sum += loss
                acc_loss_sum += acc_loss
                print('loss: ', loss)
                print('loss mean:', loss_sum / loss_num)
                print('acc loss mean:', acc_loss_sum / loss_num)

                if loss_num == 1:
                    loss_steps_sum = loss_steps
                    acc_loss_steps_sum = acc_loss_steps
                else:
                    loss_steps_sum = loss_steps_sum + loss_steps
                    acc_loss_steps_sum = acc_loss_steps_sum + acc_loss_steps

                outputs_flip = self.model(poss_flip, vels_flip, particle_r, particle_type_flip, nonk_mask_flip, self.metadata, tgt_poss_flip, tgt_vels_flip, num_rollouts=num_rollouts)

                flip_loss, flip_loss_step = self.flip_loss(outputs, outputs_flip, nonk_mask)
                flip_loss_sum += flip_loss
                if loss_num == 1:
                    flip_loss_step_sum = flip_loss_step
                else:
                    flip_loss_step_sum += flip_loss_step

                print('flip loss mean:', flip_loss_sum / loss_num)

                if self.output_fig:
                    # 1:output figures:
                    self.process_2D(tgt_poss, outputs, nonk_mask, particle_type, particle_r, batch_idx)
                    # 2:output flip figures:
                    # self.flip_process_2D(tgt_poss, tgt_poss_flip, outputs, outputs_flip, nonk_mask, particle_type, particle_r,
                    #                 batch_idx)
                    # 3:output gifs
                    # if self.dataset == "Small_Slide_Same_R" or self.dataset == "Small_Slide":
                    #     self.gif_2D(tgt_poss, outputs, nonk_mask, particle_type, particle_r, batch_idx)

        end_dt = time.time()
        print("end_datetime:", end_dt)
        print("time cost:", end_dt - start_dt, "s")

        loss_mean = loss_sum / loss_num
        acc_loss_mean = acc_loss_sum / loss_num
        flip_loss_mean = flip_loss_sum / loss_num
        loss_step_mean = loss_steps_sum / loss_num
        acc_loss_step_mean = acc_loss_steps_sum / loss_num
        flip_loss_step_mean = flip_loss_step_sum / loss_num

        print('loss mean:', loss_mean)
        print('acc loss mean:', acc_loss_mean)
        print('flip loss mean:', flip_loss_mean)
        print('loss step mean:', loss_step_mean)
        print('acc loss step mean:', acc_loss_step_mean)
        print('flip loss step mean:', flip_loss_step_mean)

        if num_rollouts > 1:
            utils.rollout_record_data(self.exp_name, loss_mean, acc_loss_mean, flip_loss_mean, loss_step_mean,
                                      acc_loss_step_mean, flip_loss_step_mean)
        else:
            utils.onestep_record_data(self.exp_name, loss_mean, acc_loss_mean, flip_loss_mean)

    def round(self, n, poss, vels):
        poss = torch.round(poss * (10 ** n)) / 10**n
        vels = torch.round(vels * (10 ** n)) / 10 ** n
        return poss, vels

    def flip_loss(self, outputs, flip_outputs, weighting):
        '''Sym_Mse'''
        flip_poss = flip_outputs['pred_poss']
        flip_poss = self.flip_poss(flip_poss)
        flip_loss = ((flip_poss - outputs['pred_poss']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        flip_loss_step = flip_loss.sum(dim=2).sum(dim=0) / torch.sum(weighting)
        flip_loss = flip_loss.mean(1).sum() / torch.sum(weighting)
        return flip_loss.item(), flip_loss_step

    def flip_poss(self, poss):
        flip_poss = copy.deepcopy(poss)
        flip_poss[:, :, 0] = self.metadata['bounds'][0][0] + self.metadata['bounds'][0][1] - flip_poss[:, :, 0]
        return flip_poss

    def loss(self, outputs, labels, weighting):
        loss = ((outputs['pred_poss'] - labels['poss']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        loss_timestep = loss.sum(dim=2).sum(dim=0) / torch.sum(weighting)
        loss = loss.mean(1).sum() / torch.sum(weighting)
        print(loss.item())

        # acc
        accn_loss = ((outputs['pred_accns'] - labels['accns']) * torch.unsqueeze(torch.unsqueeze(weighting, -1),
                                                                                 1)) ** 2
        accn_timestep = accn_loss.sum(dim=2).sum(dim=0) / torch.sum(weighting)
        accn_loss = accn_loss.mean(1).sum() / torch.sum(weighting)

        return loss.item(), loss_timestep, accn_loss.item(), accn_timestep


    def gif_2D(self, tgt_pos_seq, outputs, nonk_mask, particle_type, particle_r, batch_idx):
        particle_r = particle_r.detach().cpu().numpy()
        if self.if_sort:
            sort_index = sort_for_poss(tgt_pos_seq, outputs['pred_poss'], nonk_mask)

        bounds = self.metadata['out_bounds'].cpu().numpy()

        tgt_pos_seq = tgt_pos_seq.cpu().numpy()
        pred_pos_seq = outputs['pred_poss'].cpu().numpy()
        num_rollouts = tgt_pos_seq.shape[1]

        color = np.zeros([particle_type.shape[0], 3])

        color[particle_type.cpu() == 0] = [0, 0, 0]
        color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
        color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        if self.if_sort:
            if self.loss_max_num <= 30:
                for item in range(self.loss_max_num):
                    if item <= 10:
                        color[sort_index[item]] = [1, 0, item * 0.1]
                    elif item <= 20:
                        color[sort_index[item]] = [1 - (item - 10) * 0.1, 0, 1]
                    elif item <= 30:
                        color[sort_index[item]] = [0, (item - 20) * 0.1, 1]
                    elif item <= 40:
                        color[sort_index[item]] = [0, 1, 1 - (item - 30) * 0.1]
                    elif item <= 50:
                        color[sort_index[item]] = [(item - 40) * 0.1, 1, 0]
                    elif item <= 60:
                        color[sort_index[item]] = [1, 1 - (item - 50) * 0.1, 0]
                    else:
                        color[sort_index[item]] = [1, 0, 0]
            else:
                for item in range(self.loss_max_num):
                    color[sort_index[item]] = [0, 67 / 255, 168 / 255]

        DPI = 100
        fig = plt.figure(figsize=(12 * (bounds[0][1] - bounds[0][0]) / (bounds[1][1] - bounds[1][0]), 6), dpi=DPI)

        points1 = tgt_pos_seq[:, 0]
        ax1 = fig.add_subplot(121)

        if self.dataset == "Slide_Same_R":
            s = 3.14 * ((2.04 * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2)
            fps = 3
        elif self.dataset == "Slide":
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / 2 * (bounds[1][1] - bounds[1][0])) ** 2) * 0.0000033
            fps = 3
        else:
            s = 6.5
            fps = 10

        ax1.set_xlim(bounds[0][0], bounds[0][1])
        ax1.set_ylim(bounds[1][0], bounds[1][1])
        ax1.set_title('Ground truth')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        pts1 = ax1.scatter(points1[:, 0], points1[:, 1], c=color, s=s)

        points2 = pred_pos_seq[:, 0]
        ax2 = fig.add_subplot(122)
        ax2.set_xlim(bounds[0][0], bounds[0][1])
        ax2.set_ylim(bounds[1][0], bounds[1][1])
        ax2.set_title('Prediction')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        pts2 = ax2.scatter(points2[:, 0], points2[:, 1], c=color, s=s)

        ani = animation.FuncAnimation(fig, update_points, frames=num_rollouts,
                                      fargs=(pts1, pts2, tgt_pos_seq, pred_pos_seq))
        os.makedirs(self.output_dir, exist_ok=True)
        os.path.join(self.output_dir, str(batch_idx) + '.gif')
        ani.save(os.path.join(self.output_dir, str(batch_idx) + '.gif'), fps=fps)
        plt.close(fig)
        plt.close()

    def process_2D(self, tgt_pos_seq, outputs, nonk_mask, particle_type, particle_r, batch_idx):
        particle_r = particle_r.detach().cpu().numpy()

        tgt_pos_seq = tgt_pos_seq.cpu().numpy()
        pred_pos_seq = outputs['pred_poss'].cpu().numpy()
        num_rollouts = tgt_pos_seq.shape[1]

        color = np.zeros([particle_type.shape[0], 3])

        DPI = 300
        if self.dataset == "Slide_Same_R":
            bounds = [[0.1, 99.75], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((2.04 * 6 * DPI * 1.15 / 3 * (bounds[1][1] - bounds[1][0])) ** 2) * 0.0000000252
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Slide":
            bounds = [[0.1, 99.75], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / 2 * (bounds[1][1] - bounds[1][0])) ** 2) * 0.00000116
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Slide_Plus":
            bounds = [[0.1, 99.75], [0, 80]]
            # bounds = [[0.1, 99.9], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 2.9
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Crash":
            bounds = self.metadata['bounds'].cpu().numpy()
            rollout_list = [0, 10, 20, 30, 40, num_rollouts-1]
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 2.8
            tgt_height = tgt_pos_seq[:, 0, 1]
            color = np.array([(144 + tgt_height * (248-144) / 160) / 255, (81 + tgt_height * (206-81) / 160) / 255, (5+tgt_height*(154-5)/160) / 255])
            color = color.transpose(1, 0)
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 2] = [128 / 255, 41 / 255, 2 / 255]
        elif self.dataset == "Sand":
            bounds = self.metadata['out_bounds'].cpu().numpy()
            rollout_list = [0, 50, 100, 150, 200, 250, 300, num_rollouts-1]
            s = 3.14 * ((1 * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 0.000002
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [223 / 255, 183 / 255, 133 / 255]
        else:
            bounds = self.metadata['out_bounds'].cpu().numpy()
            rollout_list = range(num_rollouts)
            s = 6.5
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]

        for i in rollout_list:
            fig = plt.figure(figsize=(12 * (bounds[0][1] - bounds[0][0]) / (bounds[1][1] - bounds[1][0]), 6), dpi=DPI)
            tgt_pos_now = tgt_pos_seq[:, i]
            pred_pos_now = pred_pos_seq[:, i]
            points1 = tgt_pos_now
            ax1 = fig.add_subplot(121)

            ax1.set_xlim(bounds[0][0], bounds[0][1])
            ax1.set_ylim(bounds[1][0], bounds[1][1])
            ax1.set_title('Ground truth')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            pts1 = ax1.scatter(points1[:, 0], points1[:, 1], c=color, s=s)

            points2 = pred_pos_now
            ax2 = fig.add_subplot(122)
            ax2.set_xlim(bounds[0][0], bounds[0][1])
            ax2.set_ylim(bounds[1][0], bounds[1][1])
            ax2.set_title('Prediction')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            pts2 = ax2.scatter(points2[:, 0], points2[:, 1], c=color, s=s)

            plt.savefig(os.path.join(self.output_dir, str(self.dataset) + "_" + str(batch_idx) + "_" + str(i) + ".png"))

            plt.close(fig)
            plt.close()

    def flip_process_2D(self, tgt_pos_seq, tgt_poss_flip, outputs, outputs_flip, nonk_mask, particle_type, particle_r, batch_idx):
        particle_r = particle_r.detach().cpu().numpy()

        tgt_pos_seq = tgt_pos_seq.cpu().numpy()
        tgt_poss_flip = tgt_poss_flip.cpu().numpy()
        pred_pos_seq = outputs['pred_poss'].cpu().numpy()
        pred_pos_flip = outputs_flip['pred_poss'].cpu().numpy()
        num_rollouts = tgt_pos_seq.shape[1]

        color = np.zeros([particle_type.shape[0], 3])

        DPI = 300
        if self.dataset == "Slide_Same_R":
            bounds = [[0.1, 99.75], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((2.04 * 6 * DPI * 1.15 / 3 * (bounds[1][1] - bounds[1][0])) ** 2) * 0.0000000252
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Slide":
            bounds = [[0.1, 99.75], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / 2 * (bounds[1][1] - bounds[1][0])) ** 2) * 0.00000116
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Slide_Plus":
            bounds = [[0.1, 99.9], [0, 80]]
            rollout_list = range(num_rollouts)
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 2.9
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]
        elif self.dataset == "Crash":
            bounds = self.metadata['bounds'].cpu().numpy()
            rollout_list = [0, 10, 20, 30, 40, num_rollouts-1]
            s = 3.14 * ((particle_r * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 2.8
            tgt_height = tgt_pos_seq[:, 0, 1]
            color = np.array([(144 + tgt_height * (248-144) / 160) / 255, (81 + tgt_height * (206-81) / 160) / 255, (5+tgt_height*(154-5)/160) / 255])
            color = color.transpose(1, 0)
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 2] = [128 / 255, 41 / 255, 2 / 255]
        elif self.dataset == "Sand":
            bounds = self.metadata['out_bounds'].cpu().numpy()
            rollout_list = [0, 50, 100, 150, 200, 250, 300, num_rollouts-1]
            s = 3.14 * ((1 * 6 * DPI * 1.15 / (bounds[1][1] - bounds[1][0])) ** 2) * 0.000002
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [223 / 255, 183 / 255, 133 / 255]
        else:
            bounds = self.metadata['out_bounds'].cpu().numpy()
            rollout_list = range(num_rollouts)
            s = 6.5
            color[particle_type.cpu() == 0] = [0, 0, 0]
            color[particle_type.cpu() == 1] = [91 / 255, 55 / 255, 3 / 255]
            color[particle_type.cpu() == 2] = [150 / 255, 41 / 255, 41 / 255]

        for i in rollout_list:
            fig = plt.figure(figsize=(12 * (bounds[0][1] - bounds[0][0]) / (bounds[1][1] - bounds[1][0]), 12), dpi=DPI)
            tgt_pos_now = tgt_pos_seq[:, i]
            pred_pos_now = pred_pos_seq[:, i]
            tgt_pos_flip_now = tgt_poss_flip[:, i]
            pred_pos_flip_now = pred_pos_flip[:, i]
            points1 = tgt_pos_now
            ax1 = fig.add_subplot(221)
            ax1.set_xlim(bounds[0][0], bounds[0][1])
            ax1.set_ylim(bounds[1][0], bounds[1][1])
            ax1.set_title('Ground truth')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            pts1 = ax1.scatter(points1[:, 0], points1[:, 1], c=color, s=s)

            points2 = pred_pos_now
            ax2 = fig.add_subplot(223)
            ax2.set_xlim(bounds[0][0], bounds[0][1])
            ax2.set_ylim(bounds[1][0], bounds[1][1])
            ax2.set_title('Prediction')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            pts2 = ax2.scatter(points2[:, 0], points2[:, 1], c=color, s=s)

            points3 = tgt_pos_flip_now
            ax3 = fig.add_subplot(222)
            ax3.set_xlim(bounds[0][0], bounds[0][1])
            ax3.set_ylim(bounds[1][0], bounds[1][1])
            ax3.set_title('flip Ground truth')
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            pts3 = ax3.scatter(points3[:, 0], points3[:, 1], c=color, s=s)

            points4 = pred_pos_flip_now
            ax4 = fig.add_subplot(224)
            ax4.set_xlim(bounds[0][0], bounds[0][1])
            ax4.set_ylim(bounds[1][0], bounds[1][1])
            ax4.set_title('flip Prediction')
            ax4.get_xaxis().set_visible(False)
            ax4.get_yaxis().set_visible(False)
            pts4 = ax4.scatter(points4[:, 0], points4[:, 1], c=color, s=s)

            plt.savefig(os.path.join(self.output_dir, str(self.dataset) + "_flip_" + str(batch_idx) + "_" + str(i) + ".png"))

            plt.close(fig)
            plt.close()


    def flip(self, poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels, num_rollouts):
        '''Overall system flip'''
        poss_flip = copy.deepcopy(poss)
        poss_flip[:,:,0] = self.metadata['bounds'][0][0] + self.metadata['bounds'][0][1] - poss_flip[:,:,0]
        vels_flip = copy.deepcopy(vels)
        vels_flip[:,:,0] = -1 * vels_flip[:,:,0]
        tgt_poss_flip = copy.deepcopy(tgt_poss)
        tgt_poss_flip[:,:,0] = self.metadata['bounds'][0][0] + self.metadata['bounds'][0][1] - tgt_poss_flip[:,:,0]
        tgt_vels_flip = copy.deepcopy(tgt_vels)
        tgt_vels_flip[:,:,0] = -1 * tgt_vels_flip[:,:, 0]
        return poss_flip, vels_flip, particle_type, nonk_mask, tgt_poss_flip, tgt_vels_flip



def update_points(t, pts1, pts2, gt_pos, pred_pos):
    points1 = gt_pos[:, t]
    pts1.set_offsets(points1)

    points2 = pred_pos[:, t]
    pts2.set_offsets(points2)


def calculate_xy(poss):
    index1 = poss[:, 2] < 0.027
    index2 = poss[:, 1] > 0.02
    return index1 * index2
