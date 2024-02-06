import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
#sys.path.append('/mnt/hdd4T/mtz_home/code/SmoothPredictionRelease/')
from torch import random

sys.path.append(os.path.abspath('./'))
from utils import h36motion3d as datasets
from model import stage_4_fp
from  model import stage_single_fp
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from progress.bar import Bar


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size

    # 创建网络
    net_pred = stage_4_fp.MultiStageModel_fp(opt=opt)
    net_pk = stage_4_fp.MultiStageModel_itp(opt=opt)
    net = stage_4_fp.FeatConnector(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pk.to(opt.cuda_idx)
    net.to(opt.cuda_idx)

    # net_pk加载pk
    pk_model_path_len = './checkpoint/main_h36m_3d_all_in10_out25_ks10_dctn35_dropout_0.3_lr_0.005_d_model_16_no_decay/ckpt_best.pth.tar'
    # pk_model_path_len = './checkpoint/feature_main_h36m_3d_all_in10_out25_ks10_dctn35_dropout_0.3_lr_0.005_d_model_16_bn/ckpt_best.pth.tar'
    ckpt_pk = torch.load(pk_model_path_len)
    print(">>> ckpt_itp loaded epoch: {}".format(ckpt_pk['epoch']))
    net_pk.load_state_dict(ckpt_pk['state_dict'])
    net_pk.eval()

    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    optimizer = optim.Adam([{'params': net_pred.parameters()}, {'params': net.parameters()}], lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.DatasetsSmooth(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, net_pk, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch), epo)
            # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.96, epo)
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, net_pk, net, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            with torch.no_grad():
                print('>>> validation epoch: {:d}'.format(epo))
                ret_valid = run_model(net_pred, net_pk, net, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
                print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
                print('>>> testing epoch: {:d}'.format(epo))
                ret_test = run_model(net_pred, net_pk, net, is_train=3, data_loader=test_loader, opt=opt, epo=epo,)
                print('testing error: {:.3f}'.format(ret_test['#40ms']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_p3d_h36'] < err_best:
                err_best = ret_valid['m_p3d_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4_fp.MultiStageModel_fp(opt=opt)
    net_pk = stage_4_fp.MultiStageModel_itp(opt=opt)
    net = stage_4_fp.FeatConnector(opt=opt)

    net_pred.to(opt.cuda_idx)
    net_pk.to(opt.cuda_idx)
    net_pred.eval()
    net.to(opt.cuda_idx)
    net.eval()

    #load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]

    data_loader = {}
    for act in acts:
        dataset = datasets.Datasets(opt=opt, split=2, actions=act)
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)
    #do test
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test = run_model(net_pred, net_pk, net, is_train=3, data_loader=data_loader[act], opt=opt)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data

def run_model(net_pred, net_pk, net, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
        net.train()
    else:
        net_pred.eval()
        net.eval()

    l_p3d = 0
    l_p3d_5 = 0
    l_p3d_4 = 0
    l_p3d_3 = 0
    l_p3d_2 = 0
    l_p3d_1 = 0

    l_pk = 0
    l_pk_5 = 0
    l_pk_4 = 0
    l_pk_3 = 0
    l_pk_2 = 0
    l_pk_1 = 0

    if is_train <= 1:
        m_p3d_h36 = 0
        m_p3d_41 = 0
        m_p3d_42 = 0
        m_p3d_43 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1)*40
        m_p3d_h36 = np.zeros([opt.output_n])
        # # 检查前10帧的情况
        # titles = (np.array(range(opt.input_n + opt.output_n)) + 1) * 40
        # m_p3d_h36 = np.zeros([opt.input_n + opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    priv_n = opt.priv_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(data_loader))
    for i, (p3d_h36) in enumerate(data_loader):
        # [b,45,96]
        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        # [b,45,96]
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)
        input = p3d_h36[:, :, dim_used].clone()

        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1, obs_enc_4, obs_enc_3, obs_enc_2, obs_enc_1 = net_pred(input, input_n=in_n, output_n=out_n, priv_n=priv_n, itera=itera)

        # [b, 35, 22, 3]
        p3d_h36 = p3d_h36[:, :in_n + out_n]

        smooth1 = smooth(p3d_h36[:, :, dim_used],
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth2 = smooth(smooth1,
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth3 = smooth(smooth2,
                         sample_len=opt.input_n + opt.output_n,
                         kernel_size=opt.kernel_size).clone()


        # [b, 35, 22, 3]
        p3d_sup_4 = p3d_h36.clone()[:, :, dim_used][:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, len(dim_used) // 3, 3])
        p3d_sup_3 = smooth1.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, len(dim_used) // 3, 3])
        p3d_sup_2 = smooth2.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, len(dim_used) // 3, 3])
        p3d_sup_1 = smooth3.clone()[:, -out_n - in_n:].reshape(
            [-1, in_n + out_n, len(dim_used) // 3, 3])


        # 网络输出序列-用于计算mpjpe
        # [b, 25, 32, 3]
        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, in_n:in_n + out_n]
        p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, 32, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        # p3d_out_3 = p3d_h36.clone()[:, in_n:in_n + out_n]
        # p3d_out_3[:, :, dim_used] = p3d_out_all_3[:, in_n:in_n + out_n]
        # p3d_out_3[:, :, index_to_ignore] = p3d_out_3[:, :, index_to_equal]
        # p3d_out_3 = p3d_out_3.reshape([-1, out_n, 32, 3])
        #
        # p3d_out_2 = p3d_h36.clone()[:, in_n:in_n + out_n]
        # p3d_out_2[:, :, dim_used] = p3d_out_all_2[:, in_n:in_n + out_n]
        # p3d_out_2[:, :, index_to_ignore] = p3d_out_2[:, :, index_to_equal]
        # p3d_out_2 = p3d_out_2.reshape([-1, out_n, 32, 3])
        #
        # p3d_out_1 = p3d_h36.clone()[:, in_n:in_n + out_n]
        # p3d_out_1[:, :, dim_used] = p3d_out_all_1[:, in_n:in_n + out_n]
        # p3d_out_1[:, :, index_to_ignore] = p3d_out_1[:, :, index_to_equal]
        # p3d_out_1 = p3d_out_1.reshape([-1, out_n, 32, 3])

        # 网络输出序列-用于计算loss
        # [b, 45, 22, 3]
        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])

        grad_norm = 0
        if is_train == 0:
            net_pk.eval()
            # 2d joint loss:
            # [b, 35, 22, 3]
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4[:, :in_n + out_n] - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3[:, :in_n + out_n] - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2[:, :in_n + out_n] - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1[:, :in_n + out_n] - p3d_sup_1, dim=3))

            loss_p3d_all = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4

            # pk loss:
            # 结果 - [b, 45, 66]
            # 特征 - [b,d_model,22,35]
            pk_4, pk_3, pk_2, pk_1 = net_pk(input, input_n=in_n, output_n=out_n, priv_n=priv_n, itera=itera)

            # 做attention
            att_t_4, att_t_3, att_t_2, att_t_1, att_s_4, att_s_3, att_s_2, att_s_1 = net(pk_4, pk_3, pk_2, pk_1, obs_enc_4, obs_enc_3, obs_enc_2, obs_enc_1)

            # # 基于响应蒸馏
            # [b, 45, 22, 3]
            # pk_4 = pk_4.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
            # pk_3 = pk_3.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
            # pk_2 = pk_2.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
            # pk_1 = pk_1.reshape([batch_size, in_n + out_n + priv_n, len(dim_used) // 3, 3])
            # [b, 45, 22, 3]
            # loss_pk_4 = torch.mean(torch.norm(p3d_out_all_4[:, :in_n + out_n] - pk_4[:, :in_n + out_n], dim=3))
            # loss_pk_3 = torch.mean(torch.norm(p3d_out_all_3[:, :in_n + out_n] - pk_3[:, :in_n + out_n], dim=3))
            # loss_pk_2 = torch.mean(torch.norm(p3d_out_all_2[:, :in_n + out_n] - pk_2[:, :in_n + out_n], dim=3))
            # loss_pk_1 = torch.mean(torch.norm(p3d_out_all_1[:, :in_n + out_n] - pk_1[:, :in_n + out_n], dim=3))

            # # 基于特征蒸馏 - 欧几里得距离
            # # [b,d_model,22,35]
            # loss_pk_4 = torch.mean(torch.norm(pk_4 - obs_enc_4, dim=1))
            # loss_pk_3 = torch.mean(torch.norm(pk_3 - obs_enc_3, dim=1))
            # loss_pk_2 = torch.mean(torch.norm(pk_2 - obs_enc_2, dim=1))
            # loss_pk_1 = torch.mean(torch.norm(pk_1 - obs_enc_1, dim=1))

            # 基于特征蒸馏(attention) - L2
            # [b,d_model,22,35]
            loss_pk_4 = torch.mean(torch.norm(att_t_4 - att_s_4, dim=1))
            loss_pk_3 = torch.mean(torch.norm(att_t_3 - att_s_3, dim=1))
            loss_pk_2 = torch.mean(torch.norm(att_t_2 - att_s_2, dim=1))
            loss_pk_1 = torch.mean(torch.norm(att_t_1 - att_s_1, dim=1))

            # loss_pk_all = (1.2*loss_pk_4 + 1.2*loss_pk_3 + 1.2*loss_pk_2 + 0.4*loss_pk_1)/4
            loss_pk_all = (loss_pk_4 + loss_pk_3 + loss_pk_2 + loss_pk_1 )/4

            loss_all = loss_p3d_all + opt.pk_weight * loss_pk_all
            # loss_all = opt.pk_weight * loss_pk_all

            # # 查看每个stage和stage4之间的输出差异
            # mpjpe_p3d_41 = torch.mean(torch.norm(p3d_out_all_4[:, :in_n + out_n] - p3d_out_all_1[:, :in_n + out_n], dim=3))
            # mpjpe_p3d_42 = torch.mean(torch.norm(p3d_out_all_4[:, :in_n + out_n] - p3d_out_all_2[:, :in_n + out_n], dim=3))
            # mpjpe_p3d_43 = torch.mean(torch.norm(p3d_out_all_4[:, :in_n + out_n] - p3d_out_all_3[:, :in_n + out_n], dim=3))

            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d_all.cpu().data.numpy() * batch_size
            l_pk += loss_pk_all.cpu().data.numpy() * batch_size

            l_p3d_4 += loss_p3d_4.cpu().data.numpy() * batch_size
            l_p3d_3 += loss_p3d_3.cpu().data.numpy() * batch_size
            l_p3d_2 += loss_p3d_2.cpu().data.numpy() * batch_size
            l_p3d_1 += loss_p3d_1.cpu().data.numpy() * batch_size

            l_pk_4 += loss_pk_4.cpu().data.numpy() * batch_size
            l_pk_3 += loss_pk_3.cpu().data.numpy() * batch_size
            l_pk_2 += loss_pk_2.cpu().data.numpy() * batch_size
            l_pk_1 += loss_pk_1.cpu().data.numpy() * batch_size

            # m_p3d_41 += mpjpe_p3d_41.cpu().data.numpy() * batch_size
            # m_p3d_42 += mpjpe_p3d_42.cpu().data.numpy() * batch_size
            # m_p3d_43 += mpjpe_p3d_43.cpu().data.numpy() * batch_size


        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out_4[:, :out_n], dim=3))
            # # 检查输入10帧的情况
            # mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, :in_n + out_n] - p3d_out_4[:, :in_n + out_n], dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size

            # mpjpe_p3d_h36_41 = torch.mean(torch.norm(p3d_out_1[:, in_n:in_n + out_n] - p3d_out_4[:, :out_n], dim=3))
            # m_p3d_41 += mpjpe_p3d_h36_41.cpu().data.numpy() * batch_size

            # mpjpe_p3d_h36_42 = torch.mean(torch.norm(p3d_out_2[:, in_n:in_n + out_n] - p3d_out_4[:, :out_n], dim=3))
            # m_p3d_42 += mpjpe_p3d_h36_42.cpu().data.numpy() * batch_size

            # mpjpe_p3d_h36_43 = torch.mean(torch.norm(p3d_out_3[:, in_n:in_n + out_n] - p3d_out_4[:, :out_n], dim=3))
            # m_p3d_43 += mpjpe_p3d_h36_43.cpu().data.numpy() * batch_size

        else:
            # norm = torch.norm(p3d_h36[:, in_n:in_n+ out_n] - p3d_out_4[:, in_n:in_n + out_n], dim=3)
            # mean = torch.mean(norm, dim=2)
            # sum = torch.sum(mean, dim=0)
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out_4[:, :out_n], dim=3), dim=2), dim=0)
            # # 检查输入10帧的情况
            # mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, :in_n + out_n] - p3d_out_4[:, :in_n + out_n], dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        # if i % 1000 == 0:
        #     print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
        #                                                    time.time() - st, grad_norm))
        bar.suffix = '{}/{}|batch time {:.3f}s|total time{:.2f}s'.format(i + 1, len(data_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
        ret["l_p3d_4"] = l_p3d_4 / n
        ret["l_p3d_3"] = l_p3d_3 / n
        ret["l_p3d_2"] = l_p3d_2 / n
        ret["l_p3d_1"] = l_p3d_1 / n

        # ret["m_p3d_41"] = m_p3d_41 / n
        # ret["m_p3d_42"] = m_p3d_42 / n
        # ret["m_p3d_43"] = m_p3d_43 / n

        ret["l_pk"] = l_pk / n
        ret["l_pk_4"] = l_pk_4 / n
        ret["l_pk_3"] = l_pk_3 / n
        ret["l_pk_2"] = l_pk_2 / n
        ret["l_pk_1"] = l_pk_1 / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n

    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret

if __name__ == '__main__':

    option = Options().parse()

    if option.is_eval == False:
        main(opt=option)
    else:
        eval(option)
