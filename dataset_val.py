import joblib
from torch.utils.data import Dataset
from config import args
from utils.common_utils import *
from utils.ans_punct import *
import torch

if args.filter_vocb == True:
    ans_dict = load_pick('data/okvqa/ans_dic_filter_times.pickle')
else:
    ans_dict = load_pick('./data/okvqa/ans_dic.pickle')

if args.pretrain:
    image_feature = load_pick('./data/vqa_v2/vqa_img_feature_val_filter.pickle')
    val_data = load_json('./data/vqa_v2/vqa_val_filter.json')

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
    image_feature = load_pick('./data/okvqa/okvqa_img_feature_val.pickle')
    val_data = load_json('./data/okvqa/okvqa_val.json')
    wiki_entites = load_pick('data/wikidata_okvqa_val2014_topentities.pkle')
    image_captions = load_pick('data/captions/image_captions_val2014.pkl')
    sentence_transformer_wiki_encode = load_pick(
        'data/sentence_transformer/sentence_transform_encode_val2014_wiki_sentences.pickle')




class OKVQADatasetVal(Dataset):
    def __init__(self, args):
        self.val_data = val_data
        self.image_feature = image_feature
        self.wiki_entites = wiki_entites
        # self.wiki_glove_tokens = load_pick('data/wikidata_okvqa_val2014_tokenizer.pickle')
        self.ans_dic = ans_dict
        self.sentence_transformer_wiki_encode = sentence_transformer_wiki_encode
        self.image_captions = image_captions
        self.wiki_num = args.wiki_num
        self.vocab_num=len(ans_dict)
        self.caption_num = args.caption_num
        self.image_ids, self.qids, self.questions, self.answers, self.answer_dic_ids = get_info(self.val_data, self.ans_dic,
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

        # 根据img_name获取外部知识实体
        img_name = 'COCO_val2014_{}.jpg'.format(str(image_id).zfill(12))

        if args.pretrain == False:
            entities = self.wiki_entites[img_name][0][:self.wiki_num]
            sentencce_embeds = self.sentence_transformer_wiki_encode[img_name][:self.wiki_num]
            while len(entities) < self.wiki_num:
                entities = entities + self.wiki_entites[img_name][0][:(self.wiki_num - len(entities))]
                sentencce_embeds = torch.vstack(
                    [sentencce_embeds,
                     self.sentence_transformer_wiki_encode[img_name][:(self.wiki_num - sentencce_embeds.size(0))]])
                # glove_token = np.vstack((glove_token, self.wiki_glove_tokens[img_name]['tokens'][:(self.wiki_num - len(glove_token))]))
                # glove_mask = np.vstack((glove_mask, self.wiki_glove_tokens[img_name]['masks'][:(self.wiki_num - len(glove_mask))]))
            combine_entities = ['{} is a {}'.format(en[0], en[1]) for en in entities]
            image_captions = self.image_captions[img_name]
            while len(image_captions) < self.caption_num:
                image_captions = image_captions + self.image_captions[img_name][:(self.caption_num - len(image_captions))]

        else:
            sentencce_embeds = np.array([self.sentence_transformer_wiki_encode['This is a image']] * self.wiki_num)
            image_captions = self.image_captions
            combine_entities = wiki_entites

        return qid, question, answer, image_feature, spatial_feature, answer_dic_id, torch.from_numpy(
            ans_map_all_dic_score), combine_entities, image_captions, sentencce_embeds
        # return qid, question, answer, image_feature, spatial_feature, answer_dic_id, torch.from_numpy(
        #     ans_map_all_dic_score), combine_entities, image_caption, glove_token, glove_mask


def my_collate_val(batch):
    # zip(*)功能跟zip()相反 前者是拉开拉链 后者是拉上拉链
    batch = list(zip(*batch))
    # 先获取wiki知识
    knowledges = batch[7]
    # 叠加图片描述
    if args.caption_num > 0:
        for idx, d in enumerate(knowledges):
            knowledges[idx].extend(batch[8][idx])

    sentencce_embeds = torch.stack(batch[9])

    # glove_token = batch[9]  # batch_size * wiki_num * sentence_length * 词嵌入的维度
    # glove_mask = batch[10]
    # # 由于每个句子长度不一致 ，未来了统一进行glove嵌入 需要拓展pad
    # max_length = max([i.shape[1] for i in glove_token])
    # glove_token = [np.append(i, np.zeros((i.shape[0], max_length - i.shape[1]), int), axis=1) for i in glove_token]
    # glove_mask = [np.append(i, np.zeros((i.shape[0], max_length - i.shape[1]), int), axis=1) for i in glove_mask]

    res = {'id': batch[0], 'ques': batch[1], 'ans': batch[2],
           'img': batch[3], 'spatial': batch[4], 'ans_dic_ids': batch[5], 'ans_map_all_dic_score': tupple_to_tensor(batch[6]),
           'knowledges': knowledges,
           'sentencce_embeds': sentencce_embeds,
           # 'glove': (np.stack(glove_token), np.stack(glove_mask))
           }
    # del batch, glove_token, glove_mask, max_length, knowledges
    del batch, sentencce_embeds, knowledges
    return res


def get_info(val_data, ans_dic, dataset='vqa_v2'):
    image_ids = []
    qids = []
    questions = []
    answers = []
    objects = []
    answer_dic_ids = []
    for qid, item in val_data.items():
        img_id = str(item['image_id'])
        image_ids.append(img_id)
        qids.append(qid)
        questions.append(item['question'])
        answers.append(item['multi_answers'])
        m_ans_id = [ans_dic.get(i, -1) for i in item['multi_answers']]
        answer_dic_ids.append(m_ans_id)
        if dataset == 'ok_vqa':
            objects.append(item['label'])

    return image_ids, qids, questions, answers, answer_dic_ids


def tupple_to_tensor(t):
    a = None
    for i in t:
        i = i.reshape(1, -1)
        if a == None:
            a = i
        else:
            a = torch.cat((a, i), dim=0)
    return a