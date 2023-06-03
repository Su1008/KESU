# KESU
Knowledge Enhancement and Scene Understanding for Knowledge-based Visual Question Answering

This code implements a KESU model. 

## Requirement
Python>=3.8     
Pytorch==1.8.1+cu111     
transformers==4.21.1          
sentence_transformers==2.2.2


## The Dataset
The dataset is from [OK-VQA](https://okvqa.allenai.org/), and I have also provided the extracted json.
```angular2html
|-- data
   |-- okvqa
   |  |-- okvqa_train.json
   |  |-- okvqa_val.json
```
The rest of the data files can be downloaded below.
[Other Files](https://drive.google.com/file/d/1Y7IBuxZH5SPCl52DehHQqtb8ljQUAri0/view?usp=sharing)

## SBERT Process
You can use the method of this repository to get the sentence vector [repo](https://github.com/UKPLab/sentence-transformers)
And you can also use the files I have processed.
[Processed Files](https://drive.google.com/file/d/1zuCv_4tHJOwMobfI5vEksvhB561hMVnF/view?usp=sharing)

## Image Feature
The image features are provided by and downloaded from the original bottom-up attention' [repo](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)


### Models

[Our model](https://drive.google.com/file/d/1OgoljCqV5rRZeOCQCSv5l_bd7AJmEwb7/view?usp=sharing)


## Training
The following script will start training with the default hyperparameters:

```bash
$ python3 train_triplet.py --mode train --batch_size 64 --validate --comet_num 10 --caption_num 1
```