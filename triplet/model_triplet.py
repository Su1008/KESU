'''
@Project ：my_vqa 
@File ：model_second.py
@Author ：SZQ
@Date ：2022/7/13 14:06 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel, logging

# 关闭警告
logging.set_verbosity_error()

lxmert_config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
lxmert_tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
lxmert_model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', config=lxmert_config)

from core.mca import SGGA_Decoder, SGGAT_Decoder
from config import cfgs, args
from utils.gumbel_softmax import gumbel_softmax
import numpy as np
from utils.common_utils import *


class MyModel(nn.Module):
    def __init__(self, vocab_num):
        super(MyModel, self).__init__()
        # lxmert预训练模型
        self.PreLayer = lxmert_model
        self.cls_project = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))

        self.sentence_proj = nn.Linear(384, 512)

        self.l_att_proj_sim = nn.Linear(768, 1024)
        self.v_att_proj_sim = nn.Linear(768, 1024)

        self.l_att_proj_sgga = nn.Linear(768, 512)
        self.foucu_image_to_sgga = nn.Linear(768, 512)

        # self.cls_to_rel = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))
        # self.vis_to_300 = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))

        self.backbone = SGGAT_Decoder(cfgs)
        self.mca_output_softmax_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.knowledge_proj = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 300))

        self.ans_decode = nn.Embedding(vocab_num, 300)
        init.uniform_(self.ans_decode.weight.data)

        self.caption_project = nn.Sequential(nn.Linear(384, 1024), nn.ReLU(), nn.Linear(1024, 512))

    def forward(self, input, attention_mask, token_type_ids, visual_feats, visual_pos, sentence_embed, caption_embeds):
        sentence_input = self.sentence_proj(sentence_embed)

        lxmert_output = self.PreLayer(input, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      visual_feats=visual_feats, visual_pos=visual_pos)

        language_output = lxmert_output.language_output[:, 1:-1]
        vision_output = lxmert_output.vision_output
        cls = lxmert_output.pooled_output
        vision_reason = self.cls_project(cls)

        language_input = self.l_att_proj_sim(language_output)
        vision_input = self.v_att_proj_sim(vision_output)
        sim_matrix = torch.matmul(vision_input, language_input.transpose(1, 2))  # b * v_length * l_length
        kg_output_v, _ = torch.topk(sim_matrix, dim=-1, k=1)
        hard_attention_value_v = gumbel_softmax(kg_output_v.squeeze())
        head_v = (vision_output * hard_attention_value_v.unsqueeze(-1)).sum(-2)  # sum(-2)这个维度进行加权和

        focus_image_rep = self.foucu_image_to_sgga(head_v)
        language_input = self.l_att_proj_sgga(language_output)

        caption_embeds = self.caption_project(caption_embeds)
        mca_output = self.backbone(sentence_input, language_input, focus_image_rep, caption_embeds, x_mask=None,
                                   y1_mask=None, y2_mask=None, y3_mask=None)

        mca_output_proj = self.mca_output_softmax_proj(mca_output)
        mca_output_proj_soft = F.softmax(mca_output_proj, dim=1)
        mca_output_sum = torch.sum(mca_output_proj_soft * mca_output, dim=1)

        knowledge_mlp = self.knowledge_proj(mca_output_sum)


        vis_and_know = knowledge_mlp + vision_reason

        # vis_rel = self.vis_to_300(head_v) + self.cls_to_rel(cls)
        # vis_rel = self.vis_to_300(head_v) + self.caption_mlp(caption_matt_output.squeeze())
        # vis_rel = self.caption_mlp(caption_matt_output.squeeze()) + self.cls_to_rel(cls)
        # caption_mlp = self.caption_mlp(caption_matt_output.squeeze())

        # return vis_and_know, self.vis_to_300(head_v) + self.cls_to_rel(cls)
        return vis_and_know, None

    def decode_tail(self, ans_id):
        # .squeeze() 是删除维度为1的维度
        ans_id = self.ans_decode(ans_id).squeeze()
        return ans_id.squeeze()

    def cal_sim(self, anchor, most):
        sim_out = anchor.mm(most.t())  # b*分类的类别数
        return sim_out.squeeze()
