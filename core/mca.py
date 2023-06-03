from core.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
from config import cfgs


# 多头注意力机制
class MHAtt(nn.Module):

    def __init__(self, cfgs):
        super(MHAtt, self).__init__()
        self.cfgs = cfgs
        # 设置全连接层
        self.linear_v = nn.Linear(cfgs.MCA_HIDDEN_SIZE, cfgs.MCA_HIDDEN_SIZE)
        self.linear_k = nn.Linear(cfgs.MCA_HIDDEN_SIZE, cfgs.MCA_HIDDEN_SIZE)
        self.linear_q = nn.Linear(cfgs.MCA_HIDDEN_SIZE, cfgs.MCA_HIDDEN_SIZE)
        self.linear_merge = nn.Linear(cfgs.MCA_HIDDEN_SIZE, cfgs.MCA_HIDDEN_SIZE)

        self.dropout = nn.Dropout(cfgs.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        v = self.linear_v(v).view(n_batches, -1, self.cfgs.MCA_MULTI_HEAD, self.cfgs.HIDDEN_SIZE_HEAD).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.cfgs.MCA_MULTI_HEAD, self.cfgs.HIDDEN_SIZE_HEAD).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.cfgs.MCA_MULTI_HEAD, self.cfgs.HIDDEN_SIZE_HEAD).transpose(1, 2)
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.cfgs.MCA_HIDDEN_SIZE)
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        # 这里计算 Q 乘以 K的转置，除以单头注意力的维度大小
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        # 注意mask大小需要变成 : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上 len_q是query的长度  len_k是key的长度
        if mask is not None:
            # 不确定 mask 扩张对不对
            # mask=mask.unsqueeze(1).repeat(1, self.cfgs.MCA_MULTI_HEAD, 1).unsqueeze(-1).repeat(1,1,1,key.size(-2))
            mask = mask.unsqueeze(1).repeat(1, self.cfgs.MCA_MULTI_HEAD, 1).unsqueeze(-2).repeat(1, 1, query.size(-2), 1)
            scores = scores.masked_fill(mask == 0, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        # attention结果与V相乘，得到多头注意力的结果
        return torch.matmul(att_map, value)


# 前馈神经网络
class FFN(nn.Module):
    def __init__(self, cfgs):
        super(FFN, self).__init__()

        self.mlp = MLP(in_size=cfgs.MCA_HIDDEN_SIZE, mid_size=cfgs.MCA_FF_SIZE,
                       out_size=cfgs.MCA_HIDDEN_SIZE, dropout_r=cfgs.DROPOUT_R,
                       use_relu=True)

    def forward(self, x):
        return self.mlp(x)


# 自注意机制
class SA(nn.Module):
    def __init__(self, cfgs):
        super(SA, self).__init__()
        # 多头注意力机制
        self.mhatt = MHAtt(cfgs)
        # 前馈神经网络
        self.ffn = FFN(cfgs)
        self.dropout1 = nn.Dropout(cfgs.DROPOUT_R)
        self.norm1 = LayerNorm(cfgs.MCA_HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(cfgs.DROPOUT_R)
        self.norm2 = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class SGA(nn.Module):
    def __init__(self, cfgs):
        super(SGA, self).__init__()
        self.mhatt1 = MHAtt(cfgs)
        self.mhatt2 = MHAtt(cfgs)
        self.ffn = FFN(cfgs)
        self.dropout1 = nn.Dropout(cfgs.DROPOUT_R)
        self.norm1 = LayerNorm(cfgs.MCA_HIDDEN_SIZE)
        self.dropout2 = nn.Dropout(cfgs.DROPOUT_R)
        self.norm2 = LayerNorm(cfgs.MCA_HIDDEN_SIZE)
        self.dropout3 = nn.Dropout(cfgs.DROPOUT_R)
        self.norm3 = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # x应该是外部知识
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        # y是 kv，x是q
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class SGGA(nn.Module):
    def __init__(self, cfgs):
        super(SGGA, self).__init__()
        # SA
        self.mhattSA = MHAtt(cfgs)
        self.dropoutSA = nn.Dropout(cfgs.DROPOUT_R)
        self.normSA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAA
        self.mhattGAA = MHAtt(cfgs)
        self.dropoutGAA = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAB
        self.mhattGAB = MHAtt(cfgs)
        self.dropoutGAB = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAB = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # self.ffnGAA = FFN(cfgs)
        # self.dropoutGAAF = nn.Dropout(cfgs.DROPOUT_R)
        # self.normGAAF = LayerNorm(cfgs.MCA_HIDDEN_SIZE)
        #
        # self.ffnGAB = FFN(cfgs)
        # self.dropoutGABF = nn.Dropout(cfgs.DROPOUT_R)
        # self.normGABF = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        self.ffn = FFN(cfgs)
        self.dropoutUnion = nn.Dropout(cfgs.DROPOUT_R)
        self.normUnion = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

    def forward(self, x, y1, y2, x_mask, y1_mask, y2_mask):
        # SA
        x = self.normSA(x + self.dropoutSA(self.mhattSA(x, x, x, x_mask)))

        # 分支GAA
        # y是 kv，x是q
        y_x_1 = self.normGAA(x + self.dropoutGAA(self.mhattGAA(y1, y1, x, y1_mask)))

        # 分支GAB
        y_x_2 = self.normGAB(x + self.dropoutGAB(self.mhattGAB(y2, y2, x, y2_mask)))

        # 聚合
        x = y_x_1 + y_x_2
        x = self.normUnion(x + self.dropoutUnion(self.ffn(x)))
        return x


class SGGGA(nn.Module):
    def __init__(self, cfgs):
        super(SGGGA, self).__init__()
        # SA
        self.mhattSA = MHAtt(cfgs)
        self.dropoutSA = nn.Dropout(cfgs.DROPOUT_R)
        self.normSA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAA
        self.mhattGAA = MHAtt(cfgs)
        self.dropoutGAA = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAB
        self.mhattGAB = MHAtt(cfgs)
        self.dropoutGAB = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAB = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAC
        self.mhattGAC = MHAtt(cfgs)
        self.dropoutGAC = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAC = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        self.ffn = FFN(cfgs)
        self.dropoutUnion = nn.Dropout(cfgs.DROPOUT_R)
        self.normUnion = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

    def forward(self, x, y1, y2, y3, x_mask, y1_mask, y2_mask, y3_mask):
        # SA
        x = self.normSA(x + self.dropoutSA(self.mhattSA(x, x, x, x_mask)))

        # 分支GAA
        # y是 kv，x是q
        y_x_1 = self.normGAA(x + self.dropoutGAA(self.mhattGAA(y1, y1, x, y1_mask)))

        # 分支GAB
        y_x_2 = self.normGAB(x + self.dropoutGAB(self.mhattGAB(y2, y2, x, y2_mask)))

        # 分支GAC
        y_x_3 = self.normGAC(x + self.dropoutGAC(self.mhattGAC(y3, y3, x, y3_mask)))

        # 聚合
        x = y_x_1 + y_x_2 + y_x_3
        x = self.normUnion(x + self.dropoutUnion(self.ffn(x)))
        return x

class SGGAT(nn.Module):
    def __init__(self, cfgs):
        super(SGGAT, self).__init__()
        # SA
        self.mhattSA = MHAtt(cfgs)
        self.dropoutSA = nn.Dropout(cfgs.DROPOUT_R)
        self.normSA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAA
        self.mhattGAA = MHAtt(cfgs)
        self.dropoutGAA = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAA = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAB
        self.mhattGAB = MHAtt(cfgs)
        self.dropoutGAB = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAB = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        # 分支GAC
        self.mhattGAC = MHAtt(cfgs)
        self.dropoutGAC = nn.Dropout(cfgs.DROPOUT_R)
        self.normGAC = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

        self.ffn1 = FFN(cfgs)
        self.ffn2 = FFN(cfgs)
        self.ffn3 = FFN(cfgs)
        self.dropoutUnion = nn.Dropout(cfgs.DROPOUT_R)
        self.normUnion = LayerNorm(cfgs.MCA_HIDDEN_SIZE)

    def forward(self, x, y1, y2, y3, x_mask, y1_mask, y2_mask, y3_mask):
        # SA
        x = self.normSA(x + self.dropoutSA(self.mhattSA(x, x, x, x_mask)))

        y_x_1 = self.normGAA(x + self.dropoutGAA(self.mhattGAA(y1, y1, x, y1_mask)))

        y_x_2 = self.normGAB(x + self.dropoutGAB(self.mhattGAB(y2, y2, x, y2_mask)))

        y_x_3 = self.normGAC(x + self.dropoutGAC(self.mhattGAC(y3, y3, x, y3_mask)))

        q_and_v = y_x_1 + y_x_2
        q_and_c = y_x_1 + y_x_3

        q_and_v = self.ffn1(q_and_v)
        q_and_c=self.ffn2(q_and_c)

        x=q_and_v+q_and_c

        x = self.normUnion(x+ self.dropoutUnion(self.ffn3(x)))
        return x


class MCA_Decoder(nn.Module):
    def __init__(self, cfgs):
        super(MCA_Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([SGA(cfgs) for _ in range(cfgs.MAC_DECODER_LAYER)])

    def forward(self, x, y, x_mask=None, y_mask=None):
        # 这里x是外部知识嵌入/图像描述  y是问题
        for dec in self.decoder_list:
            x = dec(x, y, x_mask, y_mask)
        return x


class MCA_Encoder(nn.Module):
    def __init__(self, cfgs):
        super(MCA_Encoder, self).__init__()
        self.encoder_list = nn.ModuleList([SA(cfgs) for _ in range(cfgs.MAC_DECODER_LAYER)])

    def forward(self, x, x_mask=None):
        for enc in self.encoder_list:
            x = enc(x, x_mask)
        return x


class SGGA_Decoder(nn.Module):
    def __init__(self, cfgs):
        super(SGGA_Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([SGGA(cfgs) for _ in range(cfgs.MAC_DECODER_LAYER)])

    def forward(self, x, y1, y2, x_mask=None, y1_mask=None, y2_mask=None):
        # 这里x是外部知识嵌入  y是问题/图像描述
        for dec in self.decoder_list:
            x = dec(x, y1, y2, x_mask, y1_mask, y2_mask)
        return x


class SGGGA_Decoder(nn.Module):
    def __init__(self, cfgs):
        super(SGGGA_Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([SGGGA(cfgs) for _ in range(cfgs.MAC_DECODER_LAYER)])

    def forward(self, x, y1, y2,y3, x_mask=None, y1_mask=None, y2_mask=None,y3_mask=None):
        # 这里x是外部知识嵌入  y是问题/图像描述
        for dec in self.decoder_list:
            x = dec(x, y1, y2, y3,x_mask, y1_mask, y2_mask,y3_mask)
        return x

class SGGAT_Decoder(nn.Module):
    def __init__(self, cfgs):
        super(SGGAT_Decoder, self).__init__()
        self.decoder_list = nn.ModuleList([SGGAT(cfgs) for _ in range(cfgs.MAC_DECODER_LAYER)])

    def forward(self, x, y1, y2, y3, x_mask=None, y1_mask=None, y2_mask=None,y3_mask=None):
        # 这里x是外部知识嵌入  y是问题/图像描述
        for dec in self.decoder_list:
            x = dec(x, y1, y2, y3, x_mask, y1_mask, y2_mask,y3_mask)
        return x
