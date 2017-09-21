"""This file is based on the pytorch-tutorial GitHub repo:
   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard
   Author: Andres FR, Goethe University Frankfurt a.M.
"""

# import torch
# import torch.nn as nn
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# from torch.autograd import Variable



import tensorflow as tf
import numpy as np
import datetime
from logger import Logger
# print(help(Logger))


def make_timestamp():
    return '{:%d_%b_%Y_%Hh%Mm%Ss}'.format(datetime.datetime.now())

rand = np.random.uniform

LOGDIR = "./logs/"
LOGGER = Logger(LOGDIR)

# log two scalar functions
freq = 1/rand(10,20)
for i in range(1000):
    LOGGER.scalar_summary(LOGDIR+"xsinx", i*np.sin(i*freq), i)
    LOGGER.scalar_summary(LOGDIR+"xcosx", i*np.cos(i*freq), i)


# logger.scalar_summary(tag, value, step+1)
# tag = tag.replace('.', '/')
# logger.histo_summary(tag, to_np(value), step+1)
# logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
# logger.image_summary(tag, images, step+1)
