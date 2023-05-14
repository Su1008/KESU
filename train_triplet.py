import json, os, random, numpy as  np, torch
from config import args, cfgs

seed = args.seed
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

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from triplet.dataset_triplet import OKVQADataset,my_collate
from triplet.dataset_val_triplet import OKVQADatasetVal,my_collate_val
from torch.utils.data import DataLoader
from triplet.model_triplet import *
from utils.logger_utils import logging
from collections import Counter

device = torch.device(cfgs.DEVICE)


def train():
    train_dataset = OKVQADataset(args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=cfgs.NUM_WORKERS, collate_fn=my_collate, shuffle=True)
    # 是否在每个epoch之后测试验证集
    if args.validate:
        test_dataset = OKVQADatasetVal(args)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=cfgs.NUM_WORKERS, collate_fn=my_collate_val)

    model = MyModel(train_dataset.vocab_num).to(device)
    logging.info('当前词典数量为%d' % train_dataset.vocab_num)
    total_steps = (len(train_dataset) // (args.batch_size / torch.cuda.device_count())) * args.num_epochs \
        if len(train_dataset) % args.batch_size == 0 \
        else (len(train_dataset) // (args.batch_size / torch.cuda.device_count()) + 1) * args.num_epochs
    if args.accumulation_steps > 0:
        total_steps = total_steps / args.accumulation_steps \
            if args.accumulation_steps % total_steps == 0 else (total_steps / args.accumulation_steps) + 1

    # criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    from triplet.contrastive_loss import ContrastiveLoss
    criterion_contra = ContrastiveLoss(measure='dot', margin=1.0, max_violation=False)

    best_acc = 0
    best_epoch = 0
    # 是否加载之前模型继续训练
    if args.load_pth == "":
        start_epoch = 0
        optimizer, scheduler = set_optim(args, cfgs, model, total_steps)
    else:
        start_epoch = int(args.load_pth.split('.')[0].split("_")[-1])
        model, optimizer, scheduler, best_eval_epoch, best_eval_acc = load_check_point(args, cfgs, model, total_steps,
                                                                                       args.reset_params)
        best_epoch = best_eval_epoch
        best_acc = best_eval_acc
    # 设置多GPU
    if torch.cuda.device_count() > 1:
        logging.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
        model = nn.DataParallel(model)

    answer_candidate_tensor = torch.arange(0, train_dataset.vocab_num).view(-1, 1).long().to(device)

    for epoch in range(start_epoch, args.num_epochs):
        loss_sum = 0
        step = 0
        tqdm_data = tqdm(train_dataloader)
        model.train()
        train_pre_ans = []
        train_ground_truths = []

        

def generate_tripleid(batch_anchor, candidate):
    # cos distance   mm是矩阵乘 t是矩阵的转置
    similarity = batch_anchor.mm(candidate.t())  # b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity, k=1, dim=1, largest=True)
    return idx_1.squeeze()



def process_batch_data(batch_data):
    # 这里分词机制就是punctuation splitting and wordpiece.
    # add_special_tokens添加[CLS] [PAD] [SEP]等符号的添加
    source_seq = lxmert_tokenizer(batch_data['ques'], padding=True, return_tensors="pt", add_special_tokens=True)
    input_id = source_seq['input_ids'].to(device)
    attention_mask = source_seq['attention_mask'].to(device)
    token_type_ids = source_seq['token_type_ids'].to(device)
    # 获取图片的相关特征
    visual_feature = torch.from_numpy(np.array(batch_data['img'], dtype=float)).float().to(device)
    spatial_feature = torch.tensor(np.array(batch_data['spatial'], dtype=float)).float().to(device)

    ans_dic_ids = batch_data['ans_dic_ids']
    ans_dic_ids_tensor = torch.tensor(ans_dic_ids).long().cuda()

    sentences_embeds = batch_data['sentencce_embeds'].to(device)
    caption_embeds = batch_data['caption_embeds'].to(device)


    return input_id, attention_mask, token_type_ids, visual_feature, spatial_feature, sentences_embeds, caption_embeds


if __name__ == '__main__':

    if args.mode == 'train':
        logging.info('--------------实验参数 ------------------')
        param = {k: v for k, v in args.__dict__.items()}
        param.update(cfgs.get_dict())
        logging.info(json.dumps(param, indent=4))
        logging.info('---------------------------------------')
        train()
    elif args.mode == 'test':
        eval()
    else:
        print('程序模式输入错误，请重新输入')
