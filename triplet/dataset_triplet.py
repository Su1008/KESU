'''
@Project ：my_vqa 
@File ：dataset_triplet.py
@Author ：SZQ
@Date ：2022/7/12 15:41 
'''
from torch.utils.data import Dataset
from utils.common_utils import *
from config import args
from utils.ans_punct import *
import torch, os

if args.filter_vocb == True:
    ans_dict = load_pick('data/okvqa/ans_dic_filter_times.pickle')
else:
    ans_dict = load_pick('data/okvqa/okvqa_ans_dic_all.pickle')

if args.pretrain:
    image_feature = load_pick('./data/vqa_v2/vqa_img_feature_train_filter.pickle')
    train_data = load_json('./data/vqa_v2/vqa_train_filter.json')
    # 如果是预训练 那么字典必须叠加
    if args.filter_vocb == True:
        ans_dict.update(load_pick('data/vqa_v2/ans_dic_filter_times.pickle'))
    else:
        ans_dict.update(load_pick('data/vqa_v2/vqav2_ans_dic.pickle'))

    wiki_entites = ['This is a image'] * args.wiki_num
    image_captions = ['This is a image'] * args.caption_num
    sentence_transformer_wiki_encode = load_pick(
        'data/sentence_transformer/sentence_transform_encode_vqav2_wiki_sentences.pickle')
else:
    image_feature = load_pick('./data/okvqa/okvqa_img_feature_train.pickle')
    train_data = load_json('./data/okvqa/okvqa_train.json')
    wiki_entites = load_pick('data/wikidata_okvqa_train2014_topentities.pkle')
    image_captions = load_pick('data/captions/image_captions_train2014_union.pkl')
    comet_sentences = load_pick('data/okvqa/train_comet_sematic_sorted.pickle')
    sentence_transformer_wiki_encode = load_pick(
        'data/sentence_transformer/sentence_transform_encode_train2014_wiki_sentences.pickle')
    sentence_transformer_caption_encode = load_pick(
        'data/sentence_transformer/sentence_transform_encode_train_captions_union.pickle')
    sentence_transformer_comet_encode = load_pick(
        'data/sentence_transformer/sentence_transform_encode_train_comet.pickle')


class OKVQADataset(Dataset):

    def __init__(self, args):
        self.image_feature = image_feature
        self.train_data = train_data
        self.ans_dic = ans_dict
        self.wiki_entites = wiki_entites
        self.comet_sentences = comet_sentences
        self.sentence_transformer_wiki_encode = sentence_transformer_wiki_encode
        self.sentence_transformer_caption_encode = sentence_transformer_caption_encode
        self.sentence_transformer_comet_encode = sentence_transformer_comet_encode

        self.image_captions = image_captions
        self.vocab_num = len(self.ans_dic)
        self.aid_to_ans = {v: k for k, v in self.ans_dic.items()}
        self.wiki_num = args.wiki_num
        self.caption_num = args.caption_num
        self.comet_num = args.comet_num
        self.image_ids, self.qids, self.questions, self.answers, self.answer_dic_ids = get_info(self.train_data, self.ans_dic,
                                                                                                'ok_vqa')

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = self.qids[index]
        question = self.questions[index]
        answer = self.answers[index]
        image_id = self.image_ids[index]
        image_feature = self.image_feature[image_id]['feats']
        spatial_feature = self.image_feature[image_id]['sp_feats']
        answer_dic_id = self.answer_dic_ids[index]
        ans_map_all_dic_score = proc_ans(answer, self.ans_dic)
        img_name = 'COCO_train2014_{}.jpg'.format(str(image_id).zfill(12))

        if args.pretrain == False:
            entities = self.wiki_entites[img_name][0][:self.wiki_num]
            sentencce_embeds = self.sentence_transformer_wiki_encode[img_name][:self.wiki_num]
            while len(entities) < self.wiki_num:
                entities = entities + self.wiki_entites[img_name][0][:(self.wiki_num - len(entities))]
                sentencce_embeds = torch.vstack(
                    [sentencce_embeds,
                     self.sentence_transformer_wiki_encode[img_name][:(self.wiki_num - sentencce_embeds.size(0))]])
            comet_sen=None
            comet_embeds=None
            if self.comet_num > 0:
                comet_sen = self.comet_sentences[qid]
                comet_embeds = self.sentence_transformer_comet_encode[qid][:self.comet_num]
                while len(comet_embeds) < self.comet_num:
                    comet_sen = comet_sen + self.comet_sentences[qid][:(self.comet_num - len(comet_sen))]
                    comet_embeds = torch.vstack(
                        [comet_embeds,
                         self.sentence_transformer_comet_encode[qid][:(self.comet_num - comet_embeds.size(0))]])

            combine_entities = ['{} is a {}'.format(en[0], en[1]) for en in entities]
            image_captions = self.image_captions[img_name]  # 这是描述文本
            caption_embeds = self.sentence_transformer_caption_encode[img_name][:self.caption_num]  # 这是描述的Sbert编码
            while len(image_captions) < self.caption_num:
                image_captions = image_captions + self.image_captions[img_name][:(self.caption_num - len(image_captions))]
                caption_embeds = torch.vstack(
                    [caption_embeds,
                     self.sentence_transformer_caption_encode[img_name][:(self.caption_num - caption_embeds.size(0))]])

        else:
            sentencce_embeds = np.array([self.sentence_transformer_wiki_encode['This is a image']] * self.wiki_num)
            image_captions = self.image_captions
            combine_entities = wiki_entites

        return qid, question, answer, image_feature, spatial_feature, answer_dic_id, torch.from_numpy(
            ans_map_all_dic_score), combine_entities, image_captions, sentencce_embeds, caption_embeds, comet_sen, comet_embeds


def get_info(train_data, ans_dic, dataset='vqa_v2'):
    image_ids = []
    qids = []
    questions = []
    answers = []
    answer_dic_ids = []  # 这里应该就是答案的词典id
    for qid, item in train_data.items():
        img_id = str(item['image_id'])
        image_ids.append(img_id)
        qids.append(qid)
        questions.append(item['question'])
        # multi-answer
        if dataset == 'ok_vqa':
            answers.append(item['multi_answers'])
            m_ans_id = [ans_dic.get(i, 0) for i in item['multi_answers']]
            answer_dic_ids.append(m_ans_id)
        # single answer
        else:
            answers.append(item['answer'])
            most_ans_id = ans_dic.get(item['answer'], 0)
            answer_dic_ids.append([most_ans_id])
    return image_ids, qids, questions, answers, answer_dic_ids


def my_collate(batch):
    batch = list(zip(*batch))
    knowledges = batch[7]

    sentencce_embeds = torch.stack(batch[9])
    caption_embeds = torch.stack(batch[10])

    res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
           'img': batch[3], 'spatial': batch[4], 'ans_dic_ids': batch[5],
           'ans_map_all_dic_score': tupple_to_tensor(batch[6]),
           'knowledges': knowledges,
           'sentencce_embeds': sentencce_embeds,
           'caption_embeds': caption_embeds,
           }
    if args.comet_num > 0:
        comet_embeds = torch.stack(batch[12])
        sentencce_embeds = torch.cat((sentencce_embeds, comet_embeds),dim=1)
        res['sentencce_embeds']=sentencce_embeds
        res['comet']=batch[11]
        del comet_embeds
    del batch, knowledges, sentencce_embeds, caption_embeds
    return res


def tupple_to_tensor(t):
    a = None
    for i in t:
        i = i.reshape(1, -1)
        if a == None:
            a = i
        else:
            a = torch.cat((a, i), dim=0)
    return a
