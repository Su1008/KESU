import json
import pickle
import collections
from utils.ans_punct import prep_ans

"""这是mukea中的做法"""
word_counts = collections.Counter()
with open('../../data/okvqa/okvqa_train.json', 'r') as f:
    train_row = json.load(f)

for qid, item in train_row.items():
    ans = item['multi_answers']
    for a in ans:
        a = prep_ans(a)
        word_counts[a] += 1
#过滤出现次数 more than 2
occur_times = 0
vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > occur_times]
# vocabulary = {x: i for i, x in enumerate(vocabulary_inv[0:3500])}
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# with open('../../data/okvqa/ans_dic_filter_times_3500.pickle', 'wb') as f:
#     pickle.dump(vocabulary, f)
with open('../../data/okvqa/okvqa_ans_dic_all.pickle', 'wb') as f:
    pickle.dump(vocabulary, f)

# mcan 是针对vqa2.0的 所以存在测试集
# """mcan 中做法"""
# with open('../../data/okvqa/okvqa_val.json', 'r') as f:
#     val_row = json.load(f)
#
# # 对train和val 进行整合
# train_row.update(val_row)
# word_counts = collections.Counter()
# for qid, item in train_row.items():
#     ans = item['multi_answers']
#     for a in ans:
#         a=prep_ans(a)
#         word_counts[a] += 1
#
# vocabulary_inv = [x[0] for x in word_counts.most_common()]
# vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
#
# with open('../../data/okvqa/okvqa_ans_dic_all.pickle', 'wb') as f:
#     pickle.dump(vocabulary, f)
