import os
import re
import random
from datetime import datetime

import torch
import yaml
import socket
import getpass
import numpy as np
from config import _C as C
import pandas as pd
import torch


def onestep_record_data(filename, pos_loss, acc_loss, overturn_loss):
    # 创建一个 DataFrame
    data = {}
    data['pos loss'] = pos_loss
    data['acc loss'] = acc_loss
    data['overturn loss'] = overturn_loss
    df = pd.DataFrame(data)

    # 写入到 Excel 文件
    directory = './record/'
    if not os.path.exists(directory):
        # 创建目录
        os.makedirs(directory)

    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间为字符串
    time_string = current_time.strftime("%m-%d-%H-%M-%S")

    filename = directory + filename + '-OneStep-' + time_string + '.xlsx'
    df.to_excel(filename, index=False)
    print(f"数据已写入到 {filename} 文件。")


def rollout_record_data(filename, pos_loss, acc_loss, overturn_loss, poss_step_loss, acc_step_loss, overturn_step_loss):
    # 创建一个 DataFrame
    data = {}
    data['pos loss'] = pos_loss
    data['acc loss'] = acc_loss
    data['overturn loss'] = overturn_loss
    data['poss step loss'] = poss_step_loss.tolist()
    data['acc step loss'] = acc_step_loss.tolist()
    data['overturn step loss'] = overturn_step_loss.tolist()
    df = pd.DataFrame(data)

    # 写入到 Excel 文件
    directory = './record/'
    if not os.path.exists(directory):
        # 创建目录
        os.makedirs(directory)

    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间为字符串
    time_string = current_time.strftime("%m-%d-%H-%M-%S")

    filename = directory + filename + '-Rollout-' + time_string + '.xlsx'
    df.to_excel(filename, index=False)
    print(f"数据已写入到 {filename} 文件。")


# def rollout_record_data(filename, pos_loss, acc_loss, overturn_loss, poss_step_loss, acc_step_loss, overturn_step_loss, energy_loss_mean, pred_ke_mean,
#                                       pred_pe_mean, pred_e_mean, truth_ke_mean, truth_pe_mean, truth_e_mean):
#     # 创建一个 DataFrame
#     data = {}
#     data['pos loss'] = pos_loss
#     data['acc loss'] = acc_loss
#     data['overturn loss'] = overturn_loss
#     data['poss step loss'] = poss_step_loss.tolist()
#     data['acc step loss'] = acc_step_loss.tolist()
#     data['overturn step loss'] = overturn_step_loss.tolist()
#     data['energy loss'] = energy_loss_mean.tolist()
#     data['pred ke'] = pred_ke_mean.tolist()
#     data['pred pe'] = pred_pe_mean.tolist()
#     data['pred e'] = pred_e_mean.tolist()
#     data['truth ke'] = truth_ke_mean.tolist()
#     data['truth pe'] = truth_pe_mean.tolist()
#     data['truth e'] = truth_e_mean.tolist()
#     df = pd.DataFrame(data)
#
#     # 写入到 Excel 文件
#     directory = './record/'
#     if not os.path.exists(directory):
#         # 创建目录
#         os.makedirs(directory)
#
#     # 获取当前时间
#     current_time = datetime.now()
#     # 格式化当前时间为字符串
#     time_string = current_time.strftime("%m-%d-%H-%M-%S")
#
#     filename = directory + filename + '-Rollout-' + time_string + '.xlsx'
#     df.to_excel(filename, index=False)
#     print(f"数据已写入到 {filename} 文件。")


def vel_decomposition(v, u):  # v在u向量上的cos和sin值
    # 计算内积
    dot_product = torch.sum(v * u, dim=1)  # 沿着第二个维度求和

    # 计算投影长度
    u_norm = torch.norm(u, dim=1)  # 沿着第二个维度计算向量长度
    v_norm = torch.norm(v, dim=1)

    # 计算cos sin夹角
    cosine_similarity = torch.abs(dot_product / (v_norm * u_norm))
    sine_similarity = torch.sqrt(1 - torch.pow(cosine_similarity, 2))

    cosine_similarity[torch.isnan(cosine_similarity)] = 0
    cosine_similarity[torch.isinf(cosine_similarity)] = 0
    sine_similarity[torch.isnan(sine_similarity)] = 0
    sine_similarity[torch.isinf(sine_similarity)] = 0

    return cosine_similarity, sine_similarity


def xy_decompose(xy):
    x = xy[:, 0]
    scalar_x, direct_x = decompose(x)
    xy[:, 0] = scalar_x
    return xy, direct_x


def decompose(x):
    '''
    Decomposing vector x to obtain scalar and direction
    :param x: vector
    :return: scalar and direction
    '''
    scalar = torch.abs(x)
    # scalar = self_abs(x)
    plus = x > 0
    minus = x < 0
    zeros = (x == 0)
    direction = plus * 1 + minus * -1 + zeros * 0

    return scalar, direction


def flip_horizontally(pos, left_bound, right_bound):
    # 将x坐标进行翻转
    pos[:, :, 0] = left_bound + right_bound - pos[:, :, 0]
    return pos


def calculate_distance(pos):
    pos_1 = torch.unsqueeze(pos, dim=1)
    pos_2 = pos[None, ...]
    distance = torch.sum((pos_1 - pos_2) ** 2, dim=-1) ** 0.5
    return distance


def sort_for_poss(tgt_pos_seq, pred_pos_seq, nonk_mask):
    # 首先计算每一个粒子的位置的平均偏差
    # 根据这个偏差排序
    loss = ((pred_pos_seq - tgt_pos_seq) * torch.unsqueeze(torch.unsqueeze(nonk_mask, -1), 1)) ** 2
    loss = loss.sum(2).mean(1).cpu()  # sum(2)是计算xy两个方向偏差之和，表示粒子偏差距离；mean是计算44个时间步的平均值
    # 排序，生成倒序，从大到小的索引值
    index = np.argsort(-loss)
    return index


def lcm(a, b):
    """质因数分解"""
    p = 1
    i = 2
    while i <= min(a, b):
        if a % i == 0 and b % i == 0:
            p *= i
            a, b = a // i, b // i
        else:
            i += 1
    p = p * a * b
    return p


def ture_rollout_index(n, m):
    """生成长度为n的list，每一个元素来自（0，m），不重合"""
    numbers = []
    while len(numbers) < n:
        i = random.randint(0, m-1)
        if i not in numbers:
            numbers.append(i)
    return numbers


def caculate_config(train_set_lenth, batch_size, max_iters_times, max_iters_tired_times):
    max_iters = train_set_lenth * max_iters_times  # 放大max_iters_times倍
    max_iters_tired = train_set_lenth * max_iters_tired_times  # 放大max_iters_tired_times倍
    return max_iters // batch_size, max_iters_tired // batch_size


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)
    
def _combine_std(std_x, std_y):
  return np.sqrt(std_x**2 + std_y**2)

def update_metadata(metadata, device):
    updated_metadata = {}
    for key in metadata:
        if key == 'vel_std':
            updated_metadata[key] = _combine_std(metadata[key], C.NET.NOISE).to(device)
        elif key == 'acc_std':
            updated_metadata[key] = _combine_std(metadata[key], C.NET.NOISE).to(device)
        else:
            updated_metadata[key] = metadata[key].to(device)

    return updated_metadata

def time_diff(input_seq):
    return input_seq[:, 1:] - input_seq[:, :-1]

def get_random_walk_noise(pos_seq, idx_timestep, noise_std):
    noise_shape = (pos_seq.shape[0], pos_seq.shape[1]-1, pos_seq.shape[2])
    n_step_vel = noise_shape[1]
    # print('noise_sed:', noise_std)
    # print('n_step_vel:', n_step_vel)
    # print('type1:', type(noise_std / n_step_vel ** 0.5))
    # print('type:', type(np.random.normal(0, noise_std / n_step_vel ** 0.5, size=noise_shape)))
    acc_noise = np.random.normal(0, noise_std / n_step_vel ** 0.5, size=noise_shape).astype(np.float32)
    vel_noise = np.cumsum(acc_noise, axis=1)
    pos_noise = np.cumsum(vel_noise, axis=1)
    pos_noise = np.concatenate([np.zeros_like(pos_noise[:, :1]),
                                pos_noise], axis=1)

    return pos_noise

def get_data_root():
    hostname = socket.gethostname()
    username = getpass.getuser()
    paths_yaml_fn = 'configs/paths.yaml'
    with open(paths_yaml_fn, 'r') as f: 
        paths_config = yaml.load(f, Loader=yaml.Loader)

    for hostname_re in paths_config:
        if re.compile(hostname_re).match(hostname) is not None:
            for username_re in paths_config[hostname_re]:
                if re.compile(username_re).match(username) is not None:
                    return paths_config[hostname_re][username_re]['data_dir']

    raise Exception('No matching hostname or username in config file')
