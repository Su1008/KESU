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

        for batch_data in tqdm_data:
            step += 1
            input_id, attention_mask, token_type_ids, visual_feature, spatial_feature, sentence_embed, caption_embeds = process_batch_data(
                batch_data)

            ans_id = batch_data['ans_dic_ids']

            # vis_and_konw = model(input_id, attention_mask, token_type_ids, visual_feature, spatial_feature,
            #                               sentence_embed,caption_embeds)
            vis_and_konw, output_2 = model(input_id, attention_mask, token_type_ids, visual_feature, spatial_feature,
                                           sentence_embed, caption_embeds)

            # 查找表训练
            ans_id_tensor = torch.tensor(ans_id).view(vis_and_konw.shape[0], -1).long().to(device)  # batch*10
            if torch.cuda.device_count() > 1:
                # 词嵌入
                most = model.module.decode_tail(ans_id_tensor)
            else:
                most = model.decode_tail(ans_id_tensor)
            # p=2 指的是二范式
            most = F.normalize(most, dim=-1, p=2)

            # 候选答案训练
            if torch.cuda.device_count() > 1:
                # 送入nn.ebedding编码 然后得到候选答案 与vis_and_konw计算相似度
                answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                cls1 = model.module.cal_sim(vis_and_konw, answer_candidate_tensor_train)
                if output_2!=None:
                    cls2 = model.module.cal_sim(output_2, answer_candidate_tensor_train)
            else:
                answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                cls1 = model.cal_sim(vis_and_konw, answer_candidate_tensor_train)
                if output_2 != None:
                    cls2 = model.cal_sim(output_2, answer_candidate_tensor_train)
            vis_and_konw = F.normalize(vis_and_konw, dim=1, p=2)
            if output_2 != None:
                output_2 = F.normalize(output_2, dim=1, p=2)

            optimizer.zero_grad()  # 在计算反向传播之前将梯度置为0

            # 修改 取出现次数最多的 答案
            ans_most_id = []
            for curr_ans in ans_id:
                # 找 出现次数最多的 作为答案
                counter = Counter(curr_ans)
                most_ans = counter.most_common(1)[0][0]
                ans_most_id.append(most_ans)

            ans_id_tensor = torch.tensor(ans_most_id).long().to(device)
            # 交叉熵损失 输入两个张量 分别是 B*C 和B  其中C指的是有C个类别
            if output_2 != None:
                loss_cl = criterion_ce(cls1+cls2, ans_id_tensor)
                # loss_bce = criterion_bce(cls1+cls2,  batch_data['ans_map_all_dic_score'].to(device))
            else:
                loss_cl = criterion_ce(cls1, ans_id_tensor)
                # loss_bce = criterion_bce(cls1 , batch_data['ans_map_all_dic_score'].to(device))
            loss = 0
            # 10个答案 每个都进行损失计算
            for i in range(10):
                most_i = most[:, i, :]  # most b*10*300
                # loss+=loss_bce
                loss+=loss_cl
                loss += criterion_mse(vis_and_konw, most_i) + criterion_contra(vis_and_konw, most_i)
                if output_2 != None:
                    loss += criterion_mse(output_2, most_i) + criterion_contra(output_2, most_i)

            loss_stat = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss_stat

            # 答案生成
            with torch.no_grad():
                if torch.cuda.device_count() > 1:
                    answer_candidate_tensor_train = model.module.decode_tail(answer_candidate_tensor)
                else:
                    answer_candidate_tensor_train = model.decode_tail(answer_candidate_tensor)
                answer_candidate_tensor_train = F.normalize(answer_candidate_tensor_train, dim=1, p=2)
                if output_2 != None:
                    trip_predict = generate_tripleid_2(vis_and_konw.float(), output_2.float(),
                                                       answer_candidate_tensor_train)
                else:
                    trip_predict = generate_tripleid(vis_and_konw.float(),answer_candidate_tensor_train)

                for i, pre in enumerate(ans_id):
                    train_ground_truths.append(ans_id[i])
                    train_pre_ans.append(trip_predict[i])

            tqdm_data.set_description('Epoch %d/%d train_loss = %.4f lr = %.8f' % (
                epoch + 1, args.num_epochs, loss_stat, optimizer.param_groups[0]['lr']))
        # 计算准确率
        train_acc = cal_acc_multi(train_ground_truths, train_pre_ans)
        logging.info('Epoch %d/%d , train_acc = %.4f ,loss_sum = %.4f' % (epoch + 1, args.num_epochs, train_acc, loss_sum))
        writer_to_tensorboard(loss_sum, epoch + 1, 'loss', 'train')
        # 每个epoch之后进行验证 即测试
        if args.validate:
            logging.info(f"Validation after epoch {epoch + 1}:")
            val_acc = eval(model, test_dataloader, epoch + 1, answer_candidate_tensor)
            writer_to_tensorboard(val_acc, epoch + 1, 'acc', 'val')
            if val_acc > best_acc:
                best_epoch = epoch + 1
                best_acc = val_acc
                if val_acc > 0.375:
                    check_point_save(model, optimizer, scheduler, args.model_save_dir + 'model_for_epoch_%d.pth' % (epoch + 1),
                                     best_epoch, best_acc)
    logging.info(f'本次训练中准确率最高是 Epoch = {best_epoch} , Acc = {best_acc}')
    total = sum([param.nelement() for param in model.parameters()])
    logging.info("Number of parameter: %.2fM" % (total / 1e6))


def eval(model=None, test_dataloader=None, epoch=0, answer_candidate_tensor=None):
    if model == None and test_dataloader == None and len(args.load_pth) > 0:
        epoch = int(args.load_pth.split('.')[0].split("_")[-1])
        logging.info('加载 %s 模型进行评估' % args.load_pth)
        test_dataset = OKVQADatasetVal(args)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=cfgs.NUM_WORKERS, collate_fn=my_collate_val)
        model = MyModel(test_dataset.vocab_num).to(device)
        model.load_state_dict(
            torch.load(os.path.join(args.model_save_dir, args.load_pth), map_location=device)['state_dict'])
        answer_candidate_tensor = torch.arange(0, test_dataset.vocab_num).view(-1, 1).long().to(device)

    pre_ans = []
    ground_truths = []
    model.eval()
    eval_loss_sum = 0
    for i, batch_data_val in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_id_val, attention_mask_val, token_type_ids_val, visual_feature_val, spatial_feature_val, sentence_embed, caption_embeds= process_batch_data(
                batch_data_val)
            ans_id = batch_data_val['ans_dic_ids']

            # vis_and_konw_val= model(input_id_val, attention_mask_val, token_type_ids_val,
            #                                                      visual_feature_val,
            #                                                      spatial_feature_val, sentence_embed, caption_embeds)
            vis_and_konw_val,output_2_val= model(input_id_val, attention_mask_val, token_type_ids_val,
                                                                 visual_feature_val,
                                                                 spatial_feature_val, sentence_embed, caption_embeds)

            vis_and_konw_val = F.normalize(vis_and_konw_val, dim=1, p=2)
            if output_2_val!=None:
                output_2_val = F.normalize(output_2_val, dim=1, p=2)

            if torch.cuda.device_count() > 1:
                answer_candidate_tensor_test = model.module.decode_tail(answer_candidate_tensor)
            else:
                answer_candidate_tensor_test = model.decode_tail(answer_candidate_tensor)
            answer_candidate_tensor_test = F.normalize(answer_candidate_tensor_test, dim=1, p=2)
            if output_2_val != None:
                trip_predict = generate_tripleid_2(vis_and_konw_val.float(), output_2_val.float(),
                                                   answer_candidate_tensor_test)
            else:
                trip_predict = generate_tripleid(vis_and_konw_val.float(), answer_candidate_tensor_test)
            for i, pre in enumerate(ans_id):
                ground_truths.append(ans_id[i])
                pre_ans.append(trip_predict[i])

    # 计算准确率
    val_acc = cal_acc_multi(ground_truths, pre_ans)
    logging.info('Epoch %d/%d , eval_acc = %.4f' % (epoch, args.num_epochs, val_acc))
    return val_acc


def generate_tripleid(batch_anchor, candidate):
    # cos distance   mm是矩阵乘 t是矩阵的转置
    similarity = batch_anchor.mm(candidate.t())  # b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity, k=1, dim=1, largest=True)
    return idx_1.squeeze()


def generate_tripleid_2(batch_anchor1, batch_anchor2, candidate):
    # cos distance   mm是矩阵乘 t是矩阵的转置
    similarity = batch_anchor1.mm(candidate.t())  # b * v
    similarity2 = batch_anchor2.mm(candidate.t())  # b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity + similarity2, k=1, dim=1, largest=True)
    return idx_1.squeeze()


def generate_tripleid_3(batch_anchor1, batch_anchor2, batch_anchor3, candidate):
    # cos distance   mm是矩阵乘 t是矩阵的转置
    similarity = batch_anchor1.mm(candidate.t())  # b * v
    similarity2 = batch_anchor2.mm(candidate.t())  # b * v
    similarity3 = batch_anchor3.mm(candidate.t())  # b * v

    # cos largest:True  l2 largest:False
    prob, idx_1 = torch.topk(similarity + similarity2 + similarity3, k=1, dim=1, largest=True)
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
