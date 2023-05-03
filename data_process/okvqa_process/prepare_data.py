import json


def do(model='train'):
    row = {}
    objects_label = dict()
    with open('../../data/object/objects_vocab.txt', 'rb') as f:
        for idx, object in enumerate(f.readlines()):
            objects_label[idx] = object.lower().strip().decode()
    with open('../../data/vqa_img_object_%s.json' % model, 'rb') as f:
        image_object = json.load(f)
    with open('../../data/okvqa/raw_data/OpenEnded_mscoco_%s2014_questions.json' % model) as f:
        ques = json.load(f)
    with open('../../data/okvqa/raw_data/mscoco_%s2014_annotations.json' % model, encoding='utf-8') as f:
        annotation = json.load(f)

    ques = ques['questions']
    question_types = annotation['question_types']
    annotation = annotation['annotations']

    for q, a in zip(ques, annotation):
        question = q['question']
        image_id = q['image_id']
        objects_ids = [i[0] for i in image_object[str(image_id)]['objects']]
        origin_labels = [objects_label[id] for id in objects_ids]
        type_ = question_types[a['question_type']]
        multi_answers = []
        for ans in a['answers']:
            multi_answers.append(ans['raw_answer'])
        row[q['question_id']] = {'question': question, 'image_id': image_id, 'question_type': type_,
                                 'multi_answers': multi_answers,
                                 'label': origin_labels}

    with open('../../data/okvqa/okvqa_%s.json' % model, 'w') as f:
        json.dump(row, f, indent=4)


if __name__ == '__main__':
    do('train')
    do('val')
