# from core.net_utils import FC, MLP, LayerNorm
#
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from config import cfgs


# ------------------------------
# ---- 注意力减少模块----
#   因为经过融合之后，包含的attention weights信息是巨大的，为了获得参与的特征
# 所以经过注意力减少。
# ------------------------------
# class AttFlat(nn.Module):
#     def __init__(self, cfgs):
#         super(AttFlat, self).__init__()
#         self.cfgs = cfgs
#
#         self.mlp = MLP(in_size=cfgs.MCA_HIDDEN_SIZE, mid_size=cfgs.FLAT_MLP_SIZE,
#                        out_size=cfgs.FLAT_GLIMPSES,
#                        dropout_r=cfgs.DROPOUT_R,
#                        use_relu=True)
#
#         self.linear_merge = nn.Linear(cfgs.MCA_HIDDEN_SIZE * cfgs.FLAT_GLIMPSES, cfgs.FLAT_OUT_SIZE)
#
#     def forward(self, x, x_mask):
#         att = self.mlp(x)
#         att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
#         att = F.softmax(att, dim=1)
#
#         att_list = []
#         for i in range(self.cfgs.FLAT_GLIMPSES):
#             att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
#
#         x_atted = torch.cat(att_list, dim=1)
#         x_atted = self.linear_merge(x_atted)
#
#         return x_atted


