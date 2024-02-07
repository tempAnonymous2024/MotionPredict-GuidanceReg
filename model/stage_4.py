from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import BaseModel as BaseBlock
import utils.util as util
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.opt import Options
"""
在model1的基础上添加st_gcn,修改 bn 
"""

class MultiStageModel(Module):
    def __init__(self, opt):
        super(MultiStageModel, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size
        self.d_model = opt.d_model
        # self.seq_in = seq_in
        self.dct_n = opt.dct_n
        # ks = int((kernel_size + 1) / 2)
        assert opt.kernel_size == 10

        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features//3

        self.encoder_layer_num = 1
        self.decoder_layer_num = 2

        self.input_n = opt.input_n
        self.output_n = opt.output_n
        self.priv_n = opt.priv_n

        # self.bn1 = nn.BatchNorm1d(self.d_model * self.node_n * self.dct_n)

        self.gcn_encoder1 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.encoder_layer_num)

        self.gcn_encoder1_priv = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                  node_n=self.node_n,
                                                  seq_len=self.dct_n,
                                                  p_dropout=opt.drop_out,
                                                  num_stage=self.encoder_layer_num)

        # concat连接降维
        self.conv = nn.Conv2d(in_channels=self.d_model*2, out_channels=self.d_model, kernel_size=1)

        self.gcn_decoder_nocopy = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                  node_n=self.node_n,
                                                  seq_len=self.dct_n,
                                                  p_dropout=opt.drop_out,
                                                  num_stage=self.decoder_layer_num)

        self.gcn_decoder1 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n*2,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.decoder_layer_num)

        self.gcn_encoder2 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder2 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder3 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder3 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder4 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder4 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

    def forward(self, src, input_n=10, output_n=10, priv_n=10, itera=1):
        output_n = self.output_n
        input_n = self.input_n
        priv_n = self.priv_n

        # src [16, 45, 66]
        bs = src.shape[0]
        # [2000,512,22,20]
        dct_n = self.dct_n
        # idx = list(range(self.kernel_size)) + [self.kernel_size -1] * output_n
        idx_obs = list(range(self.kernel_size)) + [self.kernel_size -1] * (output_n + priv_n)
        idx_priv = [input_n + output_n] * (output_n + priv_n) + list(range(input_n + output_n, input_n + output_n + priv_n))
        # # input_gcn [b, 35, 66]
        # input_gcn = src[:, idx].clone()
        # input_gcn_obs [b, 45, 66]
        input_obs = src[:, idx_obs].clone()
        # input_gcn_priv [b, 45, 66]
        input_priv = src[:, idx_priv].clone()

        # dct_m, idct_m = util.get_dct_matrix(input_n + output_n)
        dct_m, idct_m = util.get_dct_matrix(input_n + output_n + priv_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)

        # [b,35,66] -> [b,66,35]
        # input_gcn_dct = torch.matmul(dct_m[:dct_n], input_gcn).permute(0, 2, 1)
        input_obs_dct = torch.matmul(dct_m[:dct_n], input_obs).permute(0, 2, 1)
        input_priv_dct = torch.matmul(dct_m[:dct_n], input_priv).permute(0, 2, 1)

        # [b,66,35]->[b,22,3,35]->[b,3,22,35]
        input_obs_dct = input_obs_dct.reshape(bs, self.node_n, -1, self.dct_n).permute(0, 2, 1, 3)
        input_priv_dct = input_priv_dct.reshape(bs, self.node_n, -1, self.dct_n).permute(0, 2, 1, 3)


        # #stage1
        # # [b,3,22,35]->[b, d_model, 22, 35]
        # latent_gcn_dct = self.gcn_encoder1(input_gcn_dct)
        # #[b,d_model,22,35] -> [b, d_model, 22, 70]
        # latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        # # [b,3,22,35]
        # output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        # # stage1_concat
        # latent_obs_dct = self.gcn_encoder1(input_obs_dct)
        # latent_priv_dct = self.gcn_encoder1_priv(input_priv_dct)
        # b, d, node_n, n = latent_obs_dct.shape
        # # 沿channel做concat
        # # [b,d_model,22,35] -> [b, 2*d_model, 22, 35]
        # latent_gcn_dct = torch.cat([latent_obs_dct, latent_priv_dct], dim=1)
        # # [b,d_model,22,35] -> [b, 2*d_model, 22*35]
        # latent_gcn_dct = latent_gcn_dct.reshape(b, 2*d, -1)
        # # 降维
        # # [b,2*d_model,22*35] -> [2*d_model, b, 22*35]
        # latent_gcn_dct = torch.transpose(latent_gcn_dct, 1, 0)
        # # [2 * d_model, b, 22*35] -> [d_model, b, 22*35]
        # latent_gcn_dct = self.conv(latent_gcn_dct)
        # # [d_model,b,22*35] -> [b, d_model, 22*35]
        # latent_gcn_dct = torch.transpose(latent_gcn_dct, 1, 0)
        # # [b,d_model,22*35] -> [b, d_model, 22,35]
        # latent_gcn_dct = latent_gcn_dct.reshape(b, d, node_n, n)
        # output_dct_1 = self.gcn_decoder_nocopy(latent_gcn_dct)[:, :, :, :dct_n]

        # stage1_element wise add
        latent_obs_dct = self.gcn_encoder1(input_obs_dct)
        latent_priv_dct = self.gcn_encoder1_priv(input_priv_dct)
        latent_gcn_dct_1 = latent_obs_dct + latent_priv_dct
        b, c, n, l = latent_gcn_dct_1.shape
        # latent_gcn_dct_1 = latent_gcn_dct_1.view(b, -1).contiguous()
        # latent_gcn_dct_1 = self.bn1(latent_gcn_dct_1).view(b, c, n, l).contiguous()
        #[b,d_model,22,35] -> [b, d_model, 22, 70]
        latent_gcn_dct = torch.cat((latent_gcn_dct_1, latent_gcn_dct_1), dim=3)
        # [b,3,22,70] -> [b,3,22,35]
        output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        #stage2
        latent_gcn_dct_2 = self.gcn_encoder2(output_dct_1)
        # latent_gcn_dct_2 = latent_gcn_dct_2.view(b, -1).contiguous()
        # latent_gcn_dct_2 = self.bn1(latent_gcn_dct_2).view(b, c, n, l).contiguous()
        # [b,d_model,22,35] -> [b, d_model, 22, 70]
        latent_gcn_dct = torch.cat((latent_gcn_dct_2, latent_gcn_dct_2), dim=3)
        output_dct_2 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        #stage3
        latent_gcn_dct_3 = self.gcn_encoder3(output_dct_2)
        # latent_gcn_dct_3 = latent_gcn_dct_3.view(b, -1).contiguous()
        # latent_gcn_dct_3 = self.bn1(latent_gcn_dct_3).view(b, c, n, l).contiguous()
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct_3, latent_gcn_dct_3), dim=3)
        output_dct_3 = self.gcn_decoder3(latent_gcn_dct)[:, :, :, :dct_n]

        #stage4
        latent_gcn_dct_4 = self.gcn_encoder4(output_dct_3)
        # latent_gcn_dct_4 = latent_gcn_dct_4.view(b, -1).contiguous()
        # latent_gcn_dct_4 = self.bn1(latent_gcn_dct_4).view(b, c, n, l).contiguous()
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct_4, latent_gcn_dct_4), dim=3)
        output_dct_4 = self.gcn_decoder4(latent_gcn_dct)[:, :, :, :dct_n]

        # [b,3,22,35] -> [b,66,35]
        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)

        # [b,35,66]->[b,45,66]
        output_1 = torch.matmul(idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1))
        output_2 = torch.matmul(idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1))
        output_3 = torch.matmul(idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1))
        output_4 = torch.matmul(idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1))

        # 标准化 - 更好地计算欧几里得距离
        # latent_gcn_dct_4 = latent_gcn_dct_4.view(b, -1).contiguous()
        # latent_gcn_dct_4 = self.bn1(latent_gcn_dct_4).view(b, c, n, l).contiguous()
        #
        # latent_gcn_dct_3 = latent_gcn_dct_3.view(b, -1).contiguous()
        # latent_gcn_dct_3 = self.bn1(latent_gcn_dct_3).view(b, c, n, l).contiguous()
        #
        # latent_gcn_dct_2 = latent_gcn_dct_2.view(b, -1).contiguous()
        # latent_gcn_dct_2 = self.bn1(latent_gcn_dct_2).view(b, c, n, l).contiguous()
        #
        # latent_gcn_dct_1 = latent_gcn_dct_1.view(b, -1).contiguous()
        # latent_gcn_dct_1 = self.bn1(latent_gcn_dct_1).view(b, c, n, l).contiguous()

        return output_4, output_3, output_2, output_1

if __name__ == '__main__':
    option = Options().parse()
    option.d_model = 64
    model = MultiStageModel(opt=option).cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    src = torch.FloatTensor(torch.randn((32, 45, 66))).cuda()
    output, att_map, zero = model(src)
