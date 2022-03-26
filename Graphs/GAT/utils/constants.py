"""
Contains constants shared across the efficient GAT implementation.
"""

import os
import enum
from torch.utils.tensorboard import SummaryWriter



# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without any
# improvement on the validation dataset (measured via accuracy/micro-F1 metric), we'll break out from the training loop.
BEST_VAL_PERF = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0


#
# PPI specific information
#
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121
