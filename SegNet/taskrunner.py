import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
os.chdir(r"\\pwrtf001.catharinazkh.local\\kliniek\\Funmilayo\SegNet_code\proseg")
import sys
from pathlib import Path
import datetime
import json
import copy
import random
device = "cuda" if torch.cuda.is_available() else "cpu"
import warnings
import monai
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
import itertools
import matplotlib.pyplot as plt

extend_path = os.path.abspath('..')
sys.path.append(extend_path)
import Utils.loader as loader
import Utils.augmentation as augmentation
import Utils.plotter as plotter
import MachineLearning.models as models
import MachineLearning.executor as executor
import MachineLearning.metrics as metrics

torch.cuda.empty_cache()

class TaskRunner():
    def __init__(self, tasks_file, folder):
        self.tasks_data = json.load(open(tasks_file))
        self.tasks = self.process_tasks_file()
        self.folder =folder
        
    def process_tasks_file(self):
        def find_combs(json_data, list_combs={}, input_key=None):
            for key, value in json_data.items():
                if isinstance(value, dict):
                    find_combs(json_data[key], list_combs)
                elif isinstance(value, list):
                    list_combs[key] = value
                    if isinstance(value[0], dict):
                        all_combinations = []
                        for i in range(len(value)):
                            combinations = []
                            possibilities = []
                            keys = []
                            for key2, value2 in value[i].items():
                                if isinstance(value2, list) or isinstance(value2, dict):
                                    possibilities.append(value2)
                                    keys.append(key2)
                            combinations = list(itertools.product(*possibilities))
                            for j in range(len(combinations)):
                                for k in range(len(combinations[j])):
                                    value[i][keys[k]] = combinations[j][k]
                                all_combinations.append(value[i].copy())
                        list_combs[key] = all_combinations
            return list_combs                       

        # Function to find all possible combinations for a dictionary of lists
        def find_combinations(data):
            combinations = []
            keys = data.keys()

            for values in itertools.product(*data.values()):
                combinations.append(dict(zip(keys, values)))

            return combinations

        # Function to recursively update JSON values
        def update_values(target, source):
            if isinstance(target, dict):
                for key, value in source.items():
                    if key in target:
                        target[key] = value
                    else:
                        for key in target:
                            update_values(target[key], source)
            return target  
        
        json_list = find_combs(self.tasks_data)
        results = find_combinations(json_list)
        tasks = []
        for i in range(len(results)):
            task = copy.deepcopy(update_values(self.tasks_data, results[i]))
            tasks.append(task)
        return tasks
    
    def read(self):
        seed=0
    
        DATA_PARAMETERS = self.tasks_data["DATA_PARAMETERS"]
        DATA_DIR = Path(DATA_PARAMETERS["DATA_DIR"])   
        IMAGE_SIZE = list(DATA_PARAMETERS["IMAGE_SIZE"].values())
        STRUCTURES = list(DATA_PARAMETERS["STRUCTURES"].values())
        EQUALIZATION_MODE = DATA_PARAMETERS["EQUALIZATION_MODE"]
        BATCH_SIZE = DATA_PARAMETERS["BATCH_SIZE"]
        THREED = DATA_PARAMETERS["THREED"]
        LOAD_DOSE = DATA_PARAMETERS["load_dose"]

        patients = [path for path in DATA_DIR.glob("*") if path.is_dir() and 1 <= int(path.name.replace('PAT', '')) <= 250]
        train_indx, test_indx = train_test_split(patients, random_state=seed, train_size=225)
        train_indx, valid_indx = train_test_split(train_indx, random_state=seed, train_size=200)
        partition = {"train": train_indx, "validation": valid_indx, "test": test_indx}
          
        print("Loading training Dataset...", end="")
        self.train_set = loader.DataSet(partition["train"], masks=STRUCTURES, size=IMAGE_SIZE, folder=self.folder, equalization_mode=EQUALIZATION_MODE, threed=THREED, load_dose=LOAD_DOSE, augment=True)
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        print("Done!")
        print("")
        print("Loading validation Dataset...", end="")
        self.valid_set = loader.DataSet(partition["validation"], masks=STRUCTURES, size=IMAGE_SIZE, folder=self.folder, equalization_mode=EQUALIZATION_MODE, threed=True, load_dose=LOAD_DOSE, augment=True)
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        print("Done!")
        print("")
        print("Loading test Dataset...", end="")
        self.test_set = loader.DataSet(partition["test"], masks=STRUCTURES, size=IMAGE_SIZE, folder=self.folder, equalization_mode=EQUALIZATION_MODE, threed=True, load_dose=LOAD_DOSE, augment=False)
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )  
        print("Done!")
    
    def run(self):
        seed=0  
        
        for train_task in self.tasks:
            # MODEL PARAMETERS
            MODEL_PARAMETERS = train_task["MODEL_PARAMETERS"]
            MODEL = MODEL_PARAMETERS["MODEL"]
            IN_CHS = MODEL_PARAMETERS["IN_CHS"]
            OUT_CHS = MODEL_PARAMETERS["OUT_CHS"]
         
            
            #torch.manual_seed(seed)
            if MODEL == "SegNet2D":
                model = models.SegNet2D(in_chs=IN_CHS, out_chs=OUT_CHS) 
            elif MODEL == "SegNet3D":
                model = models.SegNet3D(in_chs=IN_CHS, out_chs=OUT_CHS)
            elif MODEL == "DoseNet2D":
                model = models.DoseNet2D(in_chs=IN_CHS, out_chs=OUT_CHS)
            elif MODEL == "DoseNet3D":
                model = models.DoseNet3D(in_chs=IN_CHS, out_chs=OUT_CHS)
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model)
            model = model.to(device)    
            
            NOW = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            CHECKPOINTS_DIR = Path("\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\SegNet_code\proseg\SegNet3D\SR\0.13_DiceFocalLoss_03_09_2024_05_01_58")#r"\\pwrtf001.catharinazkh.local\kliniek\Funmilayo\SegNet_code\proseg") / f"{MODEL}" /f"{self.folder}"/ f"{NOW}"  #aanpassen
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINTS_DIR / "settings.json", 'w') as file:
                json.dump(train_task, file, indent=4)
            
            # METRICS
            METRICS = list(train_task["METRICS"].values())
            used_metrics = []      
            for metric in METRICS:
                if metric == "HausdorffDistance":
                    used_metrics.append(metrics.HausdorffDistance())
                elif metric == "DiceScore":
                    used_metrics.append(metrics.DiceScore())
                elif metric == "RelativeVolumeDifference":
                    used_metrics.append(metrics.RelativeVolumeDifference())
                elif metric == "AELoss":
                    used_metrics.append(metrics.AELoss())
                elif metric == "RMSELoss":
                    used_metrics.append(metrics.RMSELoss())

            # OPTIMIZER PARAMETERS
            OPTIMIZER_PARAMETERS = train_task["OPTIMIZER_PARAMETERS"]
            NUM_EPOCHS = OPTIMIZER_PARAMETERS["NUM_EPOCHS"]
            LEARNING_RATE = OPTIMIZER_PARAMETERS["LEARNING_RATE"]
            OPTIMIZER = OPTIMIZER_PARAMETERS["OPTIMIZER"]
            if OPTIMIZER["name"] == "AdamW":
                WEIGHT_DECAY = OPTIMIZER["WEIGHT_DECAY"]
                BETAS = [OPTIMIZER["BETAS"]["beta1"], OPTIMIZER["BETAS"]["beta2"]]
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
            elif OPTIMIZER["name"] == "Adam":
                WEIGHT_DECAY = OPTIMIZER["WEIGHT_DECAY"]
                BETAS = [OPTIMIZER["BETAS"]["beta1"], OPTIMIZER["BETAS"]["beta2"]]
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)    
            elif OPTIMIZER["name"] == "SGD":
                WEIGHT_DECAY = OPTIMIZER["WEIGHT_DECAY"]
                MOMENTUM = OPTIMIZER["MOMENTUM"]
                optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) 
            elif OPTIMIZER["name"] == "RMSprop":
                WEIGHT_DECAY = OPTIMIZER["WEIGHT_DECAY"]
                MOMENTUM = OPTIMIZER["MOMENTUM"]
                optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)      

            # LOSS PARAMETERS
            LOSS_PARAMETERS = train_task["LOSS_PARAMETERS"]
            LOSS_FUNCTION = LOSS_PARAMETERS["LOSS_FUNCTION"]
            EARLY_STOPPING = LOSS_PARAMETERS["EARLY_STOPPING"]
            MINIMUM_VALID_LOSS = LOSS_PARAMETERS["MINIMUM_VALID_LOSS"]

            if LOSS_FUNCTION["name"] == "DiceBCELoss":
                loss_function = metrics.DiceBCELoss()
            elif LOSS_FUNCTION["name"] == "DiceLoss":
                loss_function = metrics.DiceLoss()
            elif LOSS_FUNCTION["name"] == "DiceFocalLoss":
                GAMMA = LOSS_FUNCTION["GAMMA"]
                ALPHA = LOSS_FUNCTION["ALPHA"]
                loss_function = metrics.DiceFocalLoss(gamma=GAMMA, alpha=ALPHA)
            elif LOSS_FUNCTION["name"] == "FocalLoss":
                GAMMA = LOSS_FUNCTION["GAMMA"]
                ALPHA = LOSS_FUNCTION["ALPHA"]
                loss_function = metrics.FocalLoss(gamma=GAMMA, alpha=ALPHA)
            elif LOSS_FUNCTION["name"] == "MSELoss":
                loss_function = metrics.MSELoss()





            Executor = executor.Executor(net=model, 
                                            optimizer = optimizer,
                                            loss_function = loss_function,
                                            metrics = used_metrics,
                                            train_loader=self.train_loader,
                                            valid_loader=self.valid_loader,
                                            test_loader=self.test_loader,
                                            #augmentations=used_augmentations,
                                            label_info = self.train_set.get_info(),
                                            minimum_valid_loss = MINIMUM_VALID_LOSS,
                                            CHECKPOINTS_DIR=CHECKPOINTS_DIR,
                                            device=device,
                                            seed=seed,
                                            early_stopping=EARLY_STOPPING)

            #valid_loss = Executor.train(num_epochs=NUM_EPOCHS, display_freq=1)

            test_loss = Executor.test()
            
            #loss_name = LOSS_FUNCTION["name"]
            #CHECKPOINTS_DIR.rename(CHECKPOINTS_DIR.parent / (f"{round(valid_loss, 3)}_{loss_name}_{CHECKPOINTS_DIR.name}"))