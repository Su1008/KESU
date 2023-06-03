import argparse

import platform, os

"""
解析命令行参数和选项

name or flags - 一个命名或者一个选项字符串的列表
action - 表示该选项要执行的操作
default - 当参数未在命令行中出现时使用的值
dest - 用来指定参数的位置
type - 为参数类型，例如int
choices - 用来选择输入参数的范围。例如choice = [1, 5, 10], 表示输入参数只能为1,5 或10
help - 用来描述这个选项的作用
"""
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help='选择是训练还是测试')
parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
parser.add_argument('--filter_vocb', action='store_true', help='是否采用过滤的词典')
parser.add_argument('--reset_params', action="store_true", help='重置优化器等信息')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--model_save_dir', type=str, default='model_save/', help='model file path')
parser.add_argument("--load_pth", type=str, default="", help="To continue training, path to .pth file of saved checkpoint.")
parser.add_argument("--validate", action="store_true", help="Whether to validate on val split after every epoch.")
parser.add_argument("--accumulation_steps", type=int, default=0, help="Whether to accumulation gradient")
parser.add_argument("--seed", default=42, type=int, help="set random set")
parser.add_argument('--wiki_num', default=20, type=int, help='retrieve wiki data number')
parser.add_argument('--caption_num', default=0, type=int, help='retrieve wiki data number')
parser.add_argument('--comet_num', default=0, type=int, help='retrieve comet data number')
args = parser.parse_args()
if not os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)),args.model_save_dir)):
    os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)),args.model_save_dir))

"""  
超参数设置
"""


class Cfgs():

    def __init__(self):
        super(Cfgs, self).__init__()

        # ------------------------
        # ---- Common Params ----
        # ------------------------
        if platform.system() == 'Windows':
            self.DEVICE = 'cpu'
        else:
            self.DEVICE = 'cuda'
        self.NUM_WORKERS = 0

        # Max length of  sentences
        self.MAX_TOKEN = 16

       
        # ------------------------
        # ---- MCA Network Params ----
        # ------------------------

        # MCAN中Decoder的层数
        self.MAC_DECODER_LAYER = 6
        # Model hidden size
        # (512 as default, bigger will be a sharp increase of gpu memory usage)
        self.MCA_HIDDEN_SIZE = 512
        # Multi-head number in MCA layers
        # (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
        self.MCA_MULTI_HEAD = 8
        # FeedForwardNet size in every MCA layer
        # 一般MLP中间的基本是扩大四倍
        self.MCA_FF_SIZE = int(self.MCA_HIDDEN_SIZE * 4)
        # 多头注意力中每一层的隐藏层大小
        assert self.MCA_HIDDEN_SIZE % self.MCA_MULTI_HEAD == 0

        self.HIDDEN_SIZE_HEAD = int(self.MCA_HIDDEN_SIZE / self.MCA_MULTI_HEAD)

        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------
        # The base learning rate
        self.LR_BASE = 1e-4
        # Dropout rate for all dropout layers
        # (dropout can prevent overfitting： [Dropout: a simple way to prevent neural networks from overfitting])
        self.DROPOUT_R = 0.1
        self.WARMUP_RATE = 0.1

    def get_dict(self):
        a = {k.lower(): v for k, v in cfgs.__dict__.items() if k != 'DEVICE'}
        a.update({'device': self.DEVICE})
        return a


cfgs = Cfgs()
