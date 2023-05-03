'''
@Project ：my_vqa 
@File ：logger_utils.py
@Author ：SZQ
@Date ：2022/8/9 22:10 
'''
import logging
import time
import os


def log_creater(output_dir='logs'):
    output_dir=os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_name = '{}.txt'.format(time.strftime('%Y-%m-%d_%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler 用于写入文件
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler 这个用于输出到控制台
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler 给log添加handler
    log.addHandler(file)
    log.addHandler(stream)

    return log


logging = log_creater()
