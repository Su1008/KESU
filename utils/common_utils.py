'''
@Project ：my_vqa 
@File ：common_utils.py
@Author ：SZQ
@Date ：2022/7/11 10:31 
'''
import json
import joblib
from transformers import get_linear_schedule_with_warmup
import torch, random, os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.logger_utils import logging


def load_json(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data


def load_pick(file_path):
    with open(file_path, 'rb') as input:
        data = joblib.load(input)
    return data


def save_pick(file_path, data, model='wb'):
    with open(file_path, model) as input:
        joblib.dump(data, input)


# 计算 vqa 和okvqa
# def cal_acc_multi(ground_truth, preds):
#     all_num = len(ground_truth)
#     acc_num = 0
#     ids = []
#     temp = []
#     for i, answer_id in enumerate(ground_truth):
#         pred = preds[i]
#         cnt = 0
#         for aid in answer_id:
#             if pred == aid:
#                 cnt += 1
#         if cnt == 1:
#             acc_num += 0.3
#         elif cnt == 2:
#             acc_num += 0.6
#         elif cnt > 2:
#             acc_num += 1
#     print('正确的个数：%.1f , 总个数：%d' % (acc_num, all_num))
#     return float(acc_num) / all_num
def cal_acc_multi(ground_truth, preds):
    accs = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        human_count = answer_id.count(pred)
        acc = min(1.0, float(human_count) / 3)
        accs.append(acc)
    print('正确的个数：%.1f , 总个数：%d' % (sum(accs), len(accs)))
    return float(sum(accs)) / len(accs)


def soft_acc(grount_turth,predict):
    """
        分为两种情况
            1.若预测的字符在Ground Truth中出现 则用原有的评价标准
            2.acc= k(max(0,1- (编辑距离/真实答案的长度))) 其中系数k=词干的错误率 即 max(0,1-预测的词干的编辑距离/len(GT的词干))

    :param predict:
    :param grount_turth:
    :return:
    """
    import nltk.stem as stem
    import Levenshtein
    # 思诺博词干提取器
    snowball_stemmer = stem.SnowballStemmer('english')
    accs = []
    for idx, pred in enumerate(predict):
        gt = grount_turth[idx]
        human_count = gt.count(pred)
        if human_count > 0 or pred.isdigit():
            acc = min(1.0, float(human_count) / 3)
        else:
            acc = 0.0
            t=0
            for i, ground in enumerate(gt):
                ground=str(ground)
                if len(ground)==0:
                    continue
                orin_ed = Levenshtein.distance(ground, pred)
                # 获得词干
                pred_stem = snowball_stemmer.stem(pred)
                ground_stem = snowball_stemmer.stem(ground)
                #词干的编辑距离
                stem_ed = Levenshtein.distance(pred_stem, ground_stem)
                k = max(0, 1 - float(stem_ed / len(ground_stem)))
                tmp = k * (max(0, 1 - float(orin_ed / len(ground))))
                acc += tmp
                t += 1
            acc = float(acc / t)
        accs.append(acc)
    return float(sum(accs)) / len(accs)


# 计算krvqa
def cal_acc(ground_truth, preds, return_id=False):
    all_num = len(ground_truth)
    acc_num = 0
    ids = []
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        ids.append([i, int(pred)])
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                acc_num += 1
    if return_id:
        return acc_num / all_num, ids
    else:
        return acc_num / all_num


def setup_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 强制PyTorch使用确定性算法而不是可用的非确定性算法。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_optim(args, cfgs, model, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.LR_BASE)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfgs.WARMUP_RATE * total_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler


def load_check_point(args, cfgs, model, total_steps, reset_params=False):
    check_point = torch.load(os.path.join(args.model_save_dir, args.load_pth), map_location=torch.device(cfgs.DEVICE))
    # 加载
    model.load_state_dict(check_point['state_dict'])
    if not reset_params:
        optimizer, scheduler = set_optim(args, cfgs, model, total_steps)
        logging.info('加载 optimizer && scheduler 信息......')
        optimizer.load_state_dict(check_point["optimizer"])
        scheduler.load_state_dict(check_point["lr"])
        best_eval_epoch = check_point["best_eval_epoch"]
        best_eval_acc = check_point["best_eval_acc"]
    else:
        optimizer, scheduler = set_optim(args, cfgs, model, total_steps)
        best_eval_epoch = check_point["best_eval_epoch"]
        best_eval_acc = check_point["best_eval_acc"]
    del check_point
    return model, optimizer, scheduler, best_eval_epoch, best_eval_acc


def check_point_save(model, optimizer, scheduler, check_point_path, best_eval_epoch, best_eval_acc):
    model_to_save = model.module if hasattr(model, "module") else model
    check_point_dict = {
        'state_dict': model_to_save.state_dict(),  # 保存模型参数
        'optimizer': optimizer.state_dict(),  # 保存优化器参数
        'lr': scheduler.state_dict(),  # 保存学习率参数
        'best_eval_epoch': best_eval_epoch,
        'best_eval_acc': best_eval_acc,
    }
    torch.save(check_point_dict, check_point_path)

def save_mode_only(model,  check_point_path,best_eval_epoch, best_eval_acc):
    model_to_save = model.module if hasattr(model, "module") else model
    check_point_dict = {
        'state_dict': model_to_save.state_dict(),  # 保存模型参数
        'best_eval_epoch': best_eval_epoch,
        'best_eval_acc': best_eval_acc,
    }
    torch.save(check_point_dict, check_point_path)


def writer_to_tensorboard(data, epoch, type='loss', mode='train'):
    if mode not in ['train', 'val']:
        logging.error('日志模式输入错误。。。')
        return
    if type == 'loss':
        with SummaryWriter(f'logs/loss/{mode}') as writer:
            writer.add_scalar(f'Loss', data, epoch)
    elif type == 'acc':
        with SummaryWriter(f'logs/acc/{mode}') as writer:
            writer.add_scalar(f'Accurancy', data, epoch)


def count_sentence_vec_by_glove(glove_token_embed, glove_mask):
    """
    通过glove词向量计算句子向量
    :return:
    """
    # glove_token_embed shape->batch_size * wiki_num * length * embeddingx
    # glove_mask shape->batch_size * wiki_num * length
    wiki_num = glove_token_embed.size(1)
    # 通过glove_mask进行去pad
    result = None
    for batch_idx, batch_data in enumerate(glove_token_embed):
        tmp = None
        for i in range(0, wiki_num):
            # 找mask中1的位置 最后一个位置  然后确定长度
            len = int(torch.nonzero(glove_mask[batch_idx, i, :] == 1).squeeze()[-1]) + 1
            # 这里计算句向量 取均值//TODO
            single_sentence = (torch.sum(batch_data[i, :len], dim=0) / len).unsqueeze(0)
            tmp = single_sentence if tmp == None else torch.cat([tmp, single_sentence], dim=0)
        tmp = tmp.unsqueeze(0)
        result = tmp if result == None else torch.cat([result, tmp], dim=0)
    return result


def count_similarity(questions, entities, top_k=10):
    """
    用于检索问题和 知识句子的相似性
    :return:
    """
    if questions and entities:
        result = []
    for q, e in zip(questions, entities):
        query_embedding = similarity_sentence_model.encode(q)
        passage_embedding = similarity_sentence_model.encode(e)
        # 取前几个语义相似性的句子
        score = util.dot_score(query_embedding, passage_embedding)
        # 从大到小进行排序 (text,概率)
        sorted_resutle = sorted(list(zip(e, score.numpy().tolist()[0])), key=lambda x: x[1], reverse=True)
        remove_pro = [i[0] for i in sorted_resutle]
        result.append(remove_pro[:top_k])
    return result


def sentence_encoder(sentences):
    embeddings = []
    for sentence in sentences:
        # 这里每一个句子表示都是384个维度
        embedding = encoder_sentence_model.encode(sentence, normalize_embeddings=True)
        embeddings.append(embedding)
    return torch.tensor(np.array(embeddings))


def batch_sentences_encoder(batch_sentences):
    embeddings = []
    for sentences in batch_sentences:
        embedding = sentence_encoder(sentences)
        embeddings.append(embedding)  #
    return torch.stack(embeddings)
