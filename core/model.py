'''
@Project ：my_vqa 
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

# 加载lxmert初始化配置信息
lxmert_config = LxmertConfig.from_pretrained('unc-nlp/lxmert-base-uncased', output_attentions=True)
lxmert_tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
lxmert_model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', config=lxmert_config)

# from sentence_transformers import SentenceTransformer, util
#
# similarity_sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# encoder_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# encoder_sentence_model.max_seq_length = 64

from core.mca import MCA_Decoder, MCA_Encoder, SGGA_Decoder
from core.mca import FFN, LayerNorm, FC, MLP
from config import cfgs, args
from utils.gumbel_softmax import gumbel_softmax
import numpy as np
from utils.common_utils import *


# from data_process.glove_load import init_glove_embedding


class MyModel(nn.Module):
    def __init__(self, vocab_num):
        super(MyModel, self).__init__()
        # lxmert预训练模型
        self.PreLayer = lxmert_model
        self.cls_project = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 300))

        # 定义一个lstm用于 外部知识嵌入的表征
        # self.sentence_proj = nn.LSTM(input_size=384, hidden_size=512, batch_first=True)
        self.sentence_proj = nn.Linear(384, 512)

        # 投射到计算相似度矩阵空间
        self.l_att_proj_sim = nn.Linear(768, 1024)
        self.v_att_proj_sim = nn.Linear(768, 1024)

        self.l_att_proj_sgga = nn.Linear(768, 512)
        self.foucu_image_to_sgga = nn.Linear(768, 512)


        self.conbine_reson=nn.Sequential(nn.Linear(300, 1024), nn.ReLU(), nn.Linear(1024, 300))

        # self.sa_backbone = MCA_Encoder(cfgs)
        # self.backbone = MCA_Decoder(cfgs)
        self.backbone = SGGA_Decoder(cfgs)
        self.mca_output_softmax_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.mca_project_after_softmax = nn.Linear(512, 300)
        self.proj_norm = LayerNorm(300)
        self.vocab_proj = nn.Linear(300, vocab_num)
        # glove_vocab, glove_embeddings = init_glove_embedding('./data/glove/glove.6B.50d.txt')
        # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings).float())
        # del glove_vocab, glove_embeddings

    def forward(self, input, attention_mask, token_type_ids, visual_feats, visual_pos, sentence_embed):
        # 外部知识相关代码
        # glove_token_embed = self.embedding(glove_token)
        # # 计算词向量
        # sentence_embed = count_sentence_vec_by_glove(glove_token_embed, glove_mask)

        # self.sentence_proj.flatten_parameters()
        # sentence_input, (hn, cn) = self.sentence_proj(sentence_embed)
        sentence_input = self.sentence_proj(sentence_embed)

        lxmert_output = self.PreLayer(input, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      visual_feats=visual_feats, visual_pos=visual_pos)

        # 这里有三个输出 1:-1 是过滤【CLS】和【SEP】
        language_output = lxmert_output.language_output[:, 1:-1]  # d *768
        vision_output = lxmert_output.vision_output  # 36*768
        cls = lxmert_output.pooled_output  # batch_size *768
        # cls经过一层MLP学习 视觉信息推理
        vision_reason = self.cls_project(cls)  # batch_size *300

        language_input = self.l_att_proj_sim(language_output)
        vision_input = self.v_att_proj_sim(vision_output)
        sim_matrix = torch.matmul(vision_input, language_input.transpose(1, 2))  # b * v_length * l_length
        kg_output_v, _ = torch.topk(sim_matrix, dim=-1, k=1)
        hard_attention_value_v = gumbel_softmax(kg_output_v.squeeze())
        head_v = (vision_output * hard_attention_value_v.unsqueeze(-1)).sum(-2).unsqueeze(1)  # sum(-2)这个维度进行加权和

        focus_image_rep = self.foucu_image_to_sgga(head_v)
        language_input = self.l_att_proj_sgga(language_output)
        # mca_output = self.backbone(sentence_input, union_cat, x_mask=None,
        #                            y_mask=None)  # batch_size * 句子数量 *hidden_size
        mca_output = self.backbone(sentence_input, language_input, focus_image_rep, x_mask=None,
                                   y1_mask=None, y2_mask=None)

        # 这里的softmax 是为了均衡各个句子概率 mca_output_proj 是batch_size * n_content * MCA_HIDDEN_SIZE
        mca_output_proj = self.mca_output_softmax_proj(mca_output)

        mca_output_proj_soft = F.softmax(mca_output_proj, dim=1)
        # 这里对多个文本进行加权和
        mca_output_sum = torch.sum(mca_output_proj_soft * mca_output, dim=1)
        proj_feat = self.mca_project_after_softmax(mca_output_sum) + vision_reason
        proj_feat=self.conbine_reson(proj_feat)
        #这里多一个推理


        # 对应到词典对应的概率
        proj_feat = self.vocab_proj(self.proj_norm(proj_feat))
        return proj_feat
