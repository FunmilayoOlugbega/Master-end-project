# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:00:13 2024

@author: Gast
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.chdir(r"\\pwrtf001.catharinazkh.local\\kliniek\\Funmilayo\SegNet_code\proseg")

import sys
sys.path.append('..')
import importlib
# try:
#     importlib.reload(TaskRunner.taskrunner)
    
# except:
import TaskRunner.taskrunner as taskrunner
    
import warnings
import monai
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

train_data = ["SR", "LR", "ProSeg"]

for name in train_data:
    TaskRunner = taskrunner.TaskRunner(r"\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\SegNet_code\proseg\TaskRunner\Tasks\task_SegNet3D.json", name)
    TaskRunner.read()
    TaskRunner.run()

