#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:1', help='cuda idx')
        self.parser.add_argument('--multi', dest='multi', action='store_true',
                                 help='whether multi cards')
        self.parser.add_argument("--local_rank", type=int, default=-1)
        self.parser.add_argument('--data_dir', type=str,
                                 default='/home/ysq/MyPGBIG/h3.6m/dataset',
                                 help='path to dataset')
        self.parser.add_argument('--rep_pose_dir', type=str,
                                 default='./rep_pose/rep_pose.txt',help='path to dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint', help='path to save itp checkpoint')
        self.parser.add_argument('--ckpt_fp', type=str, default='checkpoint/fp/', help='path to save itp checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=1, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=1, help='skip rate of samples for test')
        self.parser.add_argument('--extra_info', type=str, default='', help='extra information') 

        # ===============================================================
        #                     Model options
        # ===============================================================
        # self.parser.add_argument('--input_size', type=int, default=2048, help='the input size of the neural net')
        # self.parser.add_argument('--output_size', type=int, default=85, help='the output size of the neural net')
        self.parser.add_argument('--in_features', type=int, default=66, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=64, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=10, help='past frame number')
        self.parser.add_argument('--drop_out', type=float, default=0.3, help='drop out probability')
        self.parser.add_argument('--is_ffn', type=bool, default=False, help='attention add ffn')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--encoder_n', type=int, default=6, help='encoder layer num')
        self.parser.add_argument('--decoder_n', type=int, default=6, help='decoder layer num')
        self.parser.add_argument('--rep_pose_size', type=int, default=2000, help='rep_pose_size')
        self.parser.add_argument('--updata_rate', type=float, default=0.3, help='rep pose updata_rate')
        self.parser.add_argument('--input_n', type=int, default=10, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--priv_n', type=int, default=10, help='privilege frame number')
        self.parser.add_argument('--dct_n', type=int, default=35, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0001)
        self.parser.add_argument('--alpha', type=float, default=0.4)
        self.parser.add_argument('--temp', type=float, default=7)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--pk_weight', type=float, default=1)
        self.parser.add_argument('--epoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--test_batch_size', type=int, default=256)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')
        self.parser.add_argument('--test_sample_num', type=int, default=256, help='the num ofß sample, '
                                                                                  'that sampled from test dataset'
                                                                                  '{8,256,-1(all dataset)}')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
        self._initial()
        self.opt = self.parser.parse_args()

        # if not self.opt.is_eval:
        script_name = os.path.basename(sys.argv[0])[:-3]
        if self.opt.test_sample_num == -1:
            test_sample_num = 'all'
        else:
            test_sample_num = self.opt.test_sample_num

        if self.opt.test_sample_num == -2:
            test_sample_num = '8_256_all'

        log_name = '{}_{}_in{}_out{}_dctn{}_dropout_{}_lr_{}_d_model_{}_pk_{}_L3d_all_Lpk_all'.format(
                                                                          script_name,
                                                                          test_sample_num,
                                                                          self.opt.input_n,
                                                                          self.opt.output_n,
                                                                          self.opt.dct_n,
                                                                          self.opt.drop_out,
                                                                          self.opt.lr_now,
                                                                          self.opt.d_model,
                                                                          self.opt.pk_weight
                                                                          )

        # log_name = '{}_{}_in{}_out{}_ks{}_dctn{}_dropout_{}_lr_{}_d_model_{}_no_decay'.format(
        #     script_name,
        #     test_sample_num,
        #     self.opt.input_n,
        #     self.opt.output_n,
        #     self.opt.kernel_size,
        #     self.opt.dct_n,
        #     self.opt.drop_out,
        #     self.opt.lr_now,
        #     self.opt.d_model
        #     )

        self.opt.exp = log_name
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if makedir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)

        self._print()
        # log.save_options(self.opt)
        return self.opt
