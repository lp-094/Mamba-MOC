import os
import numpy as np
from config import cfg
import torch
import datasets
from importlib import import_module
import sys


#现有版本在生成密度图时是使用的（4，15）

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
gpus = cfg.GPU_ID

if len(gpus) > 1:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus).strip("[").strip("]")
else:
    # torch.cuda.set_device(gpus[0])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------

data_mode = cfg.DATASET
datasetting = import_module(f'datasets.setting.{data_mode}')
cfg_data = datasetting.cfg_data


#------------Prepare Trainer------------
from trainer import Trainer

# 将mamba2所在的目录添加到Python搜索路径
#sys.path.append(os.path.join(os.path.dirname(__file__), "./models/Mamba"))

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]

print(cfg)
print(cfg_data)

cc_trainer = Trainer(cfg_data, pwd)
cc_trainer.forward()
