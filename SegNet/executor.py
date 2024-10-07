import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List, Union
from PIL import Image
from tqdm.auto import tqdm
import os
os.chdir(r"\\pwrtf001.catharinazkh.local\\kliniek\\Funmilayo\SegNet_code\proseg")
import sys
import random
sys.path.append(os.path.abspath('..'))
import Utils.plotter as plotter
import Utils.augmentation as augmentation
from pathlib import Path

class Executor():
    def __init__(
        self,
        net,
        optimizer,
        loss_function,
        metrics,
        train_loader,
        valid_loader,
        test_loader,
        #augmentations,
        label_info,
        CHECKPOINTS_DIR,
        minimum_valid_loss = 10000,
        device = "cuda",
        seed = 0,
        early_stopping = False
    ):
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device="cuda", abbreviated=False)
        self.net = net
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.label_info = label_info
        self.CHECKPOINTS_DIR = CHECKPOINTS_DIR
        self.minimum_valid_loss = minimum_valid_loss
        self.early_stopping = early_stopping
        self.device = device
        #self.augmentations = augmentations
        
        self.metrics = metrics
        self.metric_names = []
        self.valid_metric_scores = {}
        for metric in self.metrics:
            self.metric_names.append(metric.__class__.__name__)
        
        self.image_path = self.CHECKPOINTS_DIR / "Images"
        self.image_path.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.valid_losses = []
        self.optimizer = optimizer
        
        self.loss_function = loss_function
        self.loss_name = self.loss_function.__class__.__name__
  
        # Take random images to plot
        np.random.seed(seed)
        self.slices_to_plot = np.random.choice(np.arange(valid_loader.dataset.images.shape[2]), size=4, replace=False) 
        if self.net.__class__.__name__ in ["SegNet2D", "DoseNet2D"]:
            amount = 4
            self.augmented_slices_to_plot = np.random.choice(16, size=16, replace=False)

            indx_t = np.random.choice(np.arange(len(train_loader.dataset)), size=amount, replace=False)
            self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
            self.x_fixed_t = self.x_fixed_t.to(device)
            self.y_fixed_t = self.y_fixed_t.to(device)
            
            indx_v = np.random.choice(np.arange(len(valid_loader.dataset)), size=1, replace=False)
            self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
            self.x_fixed_v = self.x_fixed_v.to(device)
            self.y_fixed_v = self.y_fixed_v.to(device)
        else:
            amount = 1
            self.augmented_slices_to_plot = np.random.choice(np.arange(train_loader.dataset.images.shape[2]), size=16, replace=False)
            
            indx_t = np.random.choice(np.arange(0,len(train_loader.dataset)-1,2), size=amount, replace=False)
            self.x_fixed_t, self.y_fixed_t = train_loader.dataset[indx_t]
            print(self.x_fixed_t.shape)
            self.x_fixed_t = self.x_fixed_t.to(device)
            self.y_fixed_t = self.y_fixed_t.to(device)            
            
            indx_v = np.random.choice(np.arange(0,len(valid_loader.dataset)-1,2), size=amount, replace=False)
            self.x_fixed_v, self.y_fixed_v = valid_loader.dataset[indx_v]
            print(self.x_fixed_v.shape)
            self.x_fixed_v = self.x_fixed_v.to(device)
            self.y_fixed_v = self.y_fixed_v.to(device)            
            
#%%
    def train_epoch(self, epoch) -> Tuple[float]:
        self.net.train()
        epoch_loss = 0
        n = 0
        a = 0
        
        for inputs, labels in tqdm(self.train_loader, leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if epoch==1 and a==1:
                    pictures = []
                    if self.net.__class__.__name__ == "SegNet3D":
                        pictures.append(plotter.plot_slice_with_contours(inputs[0,:,self.augmented_slices_to_plot].permute(1, 0, 2, 3).cpu(), labels[0,self.augmented_slices_to_plot].cpu(), self.label_info['label_info']))            
                    elif self.net.__class__.__name__ == "SegNet2D":                
                        pictures.append(plotter.plot_slice_with_contours(inputs[self.augmented_slices_to_plot].cpu(), labels[self.augmented_slices_to_plot].cpu(), self.label_info['label_info']))
                    elif self.net.__class__.__name__ == "DoseNet3D":
                        pictures.append(plotter.plot_slice_with_dose(torch.argmax(inputs[0, :, self.augmented_slices_to_plot],dim=0).cpu(), labels[0, self.augmented_slices_to_plot].cpu(), self.label_info['label_info']))
                    elif self.net.__class__.__name__ == "DoseNet2D":
                        pictures.append(plotter.plot_slice_with_dose(torch.argmax(inputs[self.augmented_slices_to_plot],dim=1).cpu(), labels[self.augmented_slices_to_plot].cpu(), self.label_info['label_info']))           
                    else:
                        raise NotImplementedError
                    
                    stack = torch.cat(pictures).permute(0,3,1,2)
                    img_grid = make_grid(
                        stack, 
                        nrow=4, 
                        padding=12, 
                        pad_value=-1, 
                    )
                    plt.imsave(self.image_path / f"augmented_images.png", img_grid.permute(1,2,0).numpy())
            a+=1
            # Updating loss function
            self.net.zero_grad()
            outputs = self.net(inputs.float())
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            n += inputs.shape[0]
            epoch_loss += loss.item() * inputs.shape[0]
        epoch_loss = epoch_loss/n
        return epoch_loss

#%%    
    def valid_epoch(self) -> Tuple[float]:
        self.net.eval()
        scores = {}
        n = 0
        scores[self.loss_name]=0
        for inputs, labels in tqdm(self.valid_loader, leave=False):
            inputs = inputs.to(self.device)
            labels  = labels.to(self.device)  
            with torch.no_grad():
                if '2D' in self.net.__class__.__name__:
                    preds = []
                    for slice in range(inputs.shape[2]):
                        preds.append(self.net(inputs[0,:,slice, None].permute(1,0,2,3).float()))
                    outputs = torch.cat(preds, dim=0)
                    loss = self.loss_function(outputs, labels[0,:])                    
                else:
                    outputs = self.net(inputs.float())
                    loss = self.loss_function(outputs, labels)    
            n += outputs.shape[0]
            scores[self.loss_name] += loss.item() * outputs.shape[0]
            
            # Calculate metrics on validation set
            for i, metric in enumerate(self.metrics): 
                if self.net.__class__.__name__ in ["DoseNet2D", "SegNet2D"]:
                    for bn in range(outputs.shape[0]):
                        #print(f"Batch: {bn}, Output Shape: {outputs.shape}, Labels Shape: {labels.shape}")
                        if self.net.__class__.__name__ == "SegNet2D":
                            score = metric(outputs[bn, None], labels[0,bn, None])
                        else:                         
                            score = metric(outputs[bn, None], labels[0,bn,None])
                        try:
                            scores[self.metric_names[i]+"_2D"] = torch.cat((scores[self.metric_names[i]+"_2D"], score), dim=1)
                        except:
                            scores[self.metric_names[i]+"_2D"] = score
                    
                    if self.net.__class__.__name__ == "SegNet2D":     
                        score_3d = metric(outputs.permute(1,0,2,3).unsqueeze(0), labels)
                    else:     
                        score_3d = metric(outputs.permute(1,0,2,3).unsqueeze(0), labels) 
                    try:
                        scores[self.metric_names[i]+"_3D"] = torch.cat((scores[self.metric_names[i]+"_3D"], score_3d), dim=1)
                    except:
                        scores[self.metric_names[i]+"_3D"] = score_3d 
                        
                elif self.net.__class__.__name__ in ["DoseNet3D", "SegNet3D"]:
                    for bn in range(outputs.shape[0]):
                        for slice in range(outputs.shape[2]):
                            if self.net.__class__.__name__ == "SegNet3D":
                                score = metric(outputs[bn, :, slice, None].permute(1,0,2,3), labels[bn, slice, None])
                            else:
                                score = metric(outputs[bn, slice, None], labels[bn, slice, None])
                            try:
                                scores[self.metric_names[i]+"_2D"] = torch.cat((scores[self.metric_names[i]+"_2D"], score), dim=1)
                            except:
                                scores[self.metric_names[i]+"_2D"] = score
                                
                    if self.net.__class__.__name__ == "SegNet3D":     
                        score_3d = metric(outputs, labels)
                    try:
                        scores[self.metric_names[i]+"_3D"] = torch.cat((scores[self.metric_names[i]+"_3D"], score_3d), dim=1)
                    except:
                        scores[self.metric_names[i]+"_3D"] = score_3d
                        
        for key in scores.keys():
            if key != self.loss_name:
                means = []
                stds = []
                maskisnan = torch.isnan(scores[key][0])
                maskisinf = torch.isinf(scores[key][0])
                means.append(torch.mean(scores[key][0][~(maskisnan | maskisinf)]))
                stds.append(torch.std(scores[key][0][~(maskisnan | maskisinf)]))
                scores[key]=torch.stack((torch.tensor(means), torch.tensor(stds)), dim=1)
            else:
                scores[key] = scores[key]/n
        print(scores)
        return scores
    
    # save train and validation results
    def save_progress_image(self, epoch, image_path):
        with torch.no_grad():
            images = []            
            if self.net.__class__.__name__ == "SegNet2D":
                #recons_train = torch.argmax(torch.softmax(self.net(self.x_fixed_t.float()),dim=1),dim=1)
                #recons_valid = torch.argmax(torch.softmax(self.net(self.x_fixed_v.float()[0,:,self.slices_to_plot].permute(1,0,2,3)),dim=1),dim=1)
                recons_train = (self.net(self.x_fixed_t.float()) > 0).long().squeeze(1)
                recons_valid = (self.net(self.x_fixed_v.float()[0, :, self.slices_to_plot].permute(1, 0, 2, 3)) > 0).long().squeeze(1)
       
                images.append(plotter.plot_slice_with_contours(self.x_fixed_t.cpu(), self.y_fixed_t.cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_t.cpu(), recons_train.cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_v.cpu()[0, :, self.slices_to_plot].permute(1,0,2,3), self.y_fixed_v.cpu()[0, self.slices_to_plot], self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_v.cpu()[0, :, self.slices_to_plot].permute(1,0,2,3), recons_valid.cpu(), self.label_info['label_info'])) 
            elif self.net.__class__.__name__ == "SegNet3D":
                #recons_train = torch.argmax(torch.softmax(self.net(self.x_fixed_t.float()),dim=1),dim=1)
                #recons_valid = torch.argmax(torch.softmax(self.net(self.x_fixed_v.float()),dim=1),dim=1)
                recons_train = (self.net(self.x_fixed_t.float()) > 0).long().squeeze(1)
                recons_valid = (self.net(self.x_fixed_t.float()) > 0).long().squeeze(1)
                images.append(plotter.plot_slice_with_contours(self.x_fixed_t[0, :, self.slices_to_plot].permute(1, 0, 2, 3).cpu(), self.y_fixed_t[0, self.slices_to_plot].cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_t[0, :, self.slices_to_plot].permute(1, 0, 2, 3).cpu(), recons_train[0, self.slices_to_plot].cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_v[0, :, self.slices_to_plot].permute(1, 0, 2, 3).cpu(), self.y_fixed_v[0, self.slices_to_plot].cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_contours(self.x_fixed_v[0, :, self.slices_to_plot].permute(1, 0, 2, 3).cpu(), recons_valid[0, self.slices_to_plot].cpu(), self.label_info['label_info'])) 
            elif self.net.__class__.__name__ == "DoseNet2D":
                recons_train = self.net(self.x_fixed_t.float())
                recons_valid = self.net(self.x_fixed_v.float()[0,:,self.slices_to_plot].permute(1,0,2,3))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_t.cpu(),dim=1), self.y_fixed_t.cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_t.cpu(),dim=1), recons_train.detach().numpy().cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_v.cpu()[0, :, self.slices_to_plot].permute(1,0,2,3),dim=1), self.y_fixed_v.cpu()[0, self.slices_to_plot], self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_v.cpu()[0, :, self.slices_to_plot].permute(1,0,2,3),dim=1), recons_valid.detach().numpy().cpu(), self.label_info['label_info']))
            elif self.net.__class__.__name__ == "DoseNet3D":
                recons_train = self.net(self.x_fixed_t.float())
                recons_valid = self.net(self.x_fixed_v.float())
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_t[0, :, self.slices_to_plot],dim=0).cpu(), self.y_fixed_t[0, self.slices_to_plot].cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_t[0, :, self.slices_to_plot],dim=0).cpu(), recons_train[0, self.slices_to_plot].detach().numpy().cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_v[0, :, self.slices_to_plot],dim=0).cpu(), self.y_fixed_v[0, self.slices_to_plot].cpu(), self.label_info['label_info']))
                images.append(plotter.plot_slice_with_dose(torch.argmax(self.x_fixed_v[0, :, self.slices_to_plot],dim=0).cpu(), recons_valid[0, self.slices_to_plot].detach().numpy().cpu(), self.label_info['label_info']))
            else:
                raise NotImplementedError                    

            stack = torch.cat(images).permute(0,3,1,2)
            img_grid = make_grid(
                stack, 
                nrow=4, 
                padding=12, 
                pad_value=-1, 
            )
            plt.imsave(image_path / f"epoch_{epoch}.png", img_grid.permute(1,2,0).numpy())
            
    def post_processing(self, image_path):
        def extract_evenly_spaced_random_items(lst, num_items):
            if num_items >= len(lst):
                return lst

            num_items -= 2  # Exclude first and last item from random selection
            interval = (len(lst) - 2) / num_items
            extracted_items = [lst[0]]  # Include first item

            for i in range(num_items):
                index = int((i * interval) + 1)  # Start from index 1 (second item)
                extracted_items.append(lst[index])

            extracted_items.append(lst[-1])  # Include last item
            return extracted_items
        #Create GIF
        frame_duration = 300
        images = []
        for filename in sorted(os.listdir(image_path)):
            if filename.startswith("epoch") and filename.endswith(".png"):
                images.append(Image.open(os.path.join(image_path, filename)))
        images = extract_evenly_spaced_random_items(images, 10)
        gif_filename = image_path / "progress_loop.gif"
        images[0].save(gif_filename,
                       save_all=True,
                       append_images=images[1:],
                       duration=frame_duration,
                       loop=0)    
#%% 
    def train( self, num_epochs : int, display_freq  : int = 1):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        no_increase = 0
        
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_epoch(epoch)
            valid_outcome = self.valid_epoch()
            
            valid_loss = valid_outcome[self.loss_name]
            
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            
            for metric in valid_outcome:
                if metric!=self.loss_name:
                    try:
                        self.valid_metric_scores[metric] = torch.cat((self.valid_metric_scores[metric], valid_outcome[metric].unsqueeze(0)), dim=0)
                    except:
                        self.valid_metric_scores[metric] = valid_outcome[metric].unsqueeze(0)

            self.scheduler.step(valid_loss) 
            if self.scheduler.get_last_lr()[0] < self.optimizer.param_groups[0]['lr']:
                print("Learning rate reduced!")

            msg = f"Epoch #{epoch:03d}: "
            msg = msg + f"{self.loss_name}/train = {train_loss:.3f}, "
            msg = msg[:-2] + " | "
            msg = msg + f"{self.loss_name}/valid = {valid_loss:.3f}, "
            msg = msg[:-2]
            print(msg)
            
            
            for key in valid_outcome:
                if key!=self.loss_name:
                    plt.clf()
                    maskisnan=torch.isnan(self.valid_metric_scores[key][:, 0, 0])
                    maskisinf=torch.isinf(self.valid_metric_scores[key][:, 0, 0])
                    filtered_tensor = self.valid_metric_scores[key][:, 0, 0][~(maskisnan | maskisinf)]
                    indices = torch.arange(self.valid_metric_scores[key].shape[0])+1
                    filtered_indices = indices[~(maskisnan | maskisinf)]
                    
                    if self.net.__class__.__name__ in ["SegNet2D", "SegNet3D"]:
                            plt.plot(filtered_indices, filtered_tensor, label=self.label_info['label_info'][1][0], color=np.array(self.label_info['label_info'][1][1])/255)
                    else:
                        plt.plot(filtered_indices, filtered_tensor, label=key)
                    plt.xlabel('Number of epochs')
                    plt.ylabel(f'{key}')
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    maskisnan=torch.isnan(self.valid_metric_scores[key][:])
                    maskisinf=torch.isinf(self.valid_metric_scores[key][:])
                    filtered_tensor = self.valid_metric_scores[key][:][~(maskisnan | maskisinf)]
                    if key!= 'RelativeVolumeDifference_3D':
                        if filtered_tensor.numel() > 0:
                            max_value = torch.max(filtered_tensor)
                            plt.ylim(0, max_value + max_value * 0.1)
                        else:
                            plt.ylim(bottom=0)#, top= torch.max(self.valid_metric_scores[key][:][~(maskisnan | maskisinf)]) + torch.max(self.valid_metric_scores[key][:][~(maskisnan | maskisinf)]) * 0.1)
                    plt.legend()
                    plot_name = self.CHECKPOINTS_DIR / f"{key}.png"
                    plt.savefig(plot_name, dpi=200)  
            
            plt.clf()
            plt.plot(range(1, len(self.train_losses)+1), self.train_losses, label="Training loss")
            plt.plot(range(1, len(self.train_losses)+1), self.valid_losses, label="Validation loss")
            plt.xlabel('Number of epochs')
            plt.ylabel(f'{self.loss_name}')
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.ylim(0, max(self.valid_losses) + max(self.valid_losses)*0.1)
            plt.legend()
            plot_name = self.CHECKPOINTS_DIR / f"{self.loss_name}.png"
            plt.savefig(plot_name, dpi=200)
            
            np.savetxt(self.CHECKPOINTS_DIR / f"{self.loss_name}_train_losses.csv", self.train_losses, delimiter=",")
            np.savetxt(self.CHECKPOINTS_DIR / f"{self.loss_name}_validation_losses.csv", self.valid_losses, delimiter=",")
            print(f"Best loss at Epoch #{self.valid_losses.index(min(self.valid_losses))+1:03d} with {self.loss_name}/valid = {min(self.valid_losses)}")
            
            for metric in valid_outcome:
                if metric!=self.loss_name:
                    np.savetxt(self.CHECKPOINTS_DIR / f"{metric}_valid_scores.csv", self.valid_metric_scores[metric][:, 0, 0], delimiter=",")
                    
            if (epoch + 1) % display_freq == 0:
                self.save_progress_image(epoch, self.image_path)
            
            if valid_loss < (self.minimum_valid_loss):
                no_increase = 0
                self.minimum_valid_loss = valid_loss
                torch.save(self.net.state_dict(), self.CHECKPOINTS_DIR / "best_model.pth")
            elif self.early_stopping:
                no_increase +=1
                if no_increase > 9:
                    print(f"Training stopped at Epoch #{epoch:03d} with {self.loss_name}/valid = {valid_loss:.3f} due to early stopping!")
                    break
            torch.save(self.net.state_dict(), self.CHECKPOINTS_DIR / "last_model.pth")

        self.post_processing(self.image_path)
        return self.minimum_valid_loss  
 
    #%%
    def test(self):
        self.net.load_state_dict(torch.load(self.CHECKPOINTS_DIR / "best_model.pth"))
        self.net.eval()
        n=0
        self.test_scores = {}
        self.test_scores[self.loss_name]=0
        for inputs, labels in tqdm(self.test_loader, leave=False):  
            inputs = inputs.to(self.device)
            labels  = labels.to(self.device)            
            with torch.no_grad():
                if '2D' in self.net.__class__.__name__:
                    preds = []
                    for slice in range(inputs.shape[2]):
                        preds.append(self.net(inputs[0,:,slice, None].permute(1,0,2,3).float()))
                    outputs = torch.cat(preds, dim=0)
                    loss = self.loss_function(outputs, labels[0,:])                       
                else:
                    outputs = self.net(inputs.float())
                    loss = self.loss_function(outputs, labels)        
            n += outputs.shape[0]
            self.test_scores[self.loss_name] += loss.item() * outputs.shape[0]
            for i, metric in enumerate(self.metrics): 
                if self.net.__class__.__name__ in ["DoseNet2D", "SegNet2D"]:
                    for bn in range(outputs.shape[0]):
                        if self.net.__class__.__name__ == "SegNet2D":                          
                            score = metric(outputs[bn, None], labels[0,bn, None])
                        else:                         
                            score = metric(outputs[bn, None], labels[0,bn, None])
                        try:
                            self.test_scores[self.metric_names[i]+"_2D"] = torch.cat((self.test_scores[self.metric_names[i]+"_2D"], score), dim=1)
                        except:
                            self.test_scores[self.metric_names[i]+"_2D"] = score

                    if self.net.__class__.__name__ == "SegNet2D":     
                        score_3d = metric(outputs.permute(1,0,2,3).unsqueeze(0), labels)
                    else:     
                        score_3d = metric(outputs.unsqueeze(0), labels) 
                    try:
                        self.test_scores[self.metric_names[i]+"_3D"] = torch.cat((self.test_scores[self.metric_names[i]+"_3D"], score_3d), dim=1)
                    except:
                        self.test_scores[self.metric_names[i]+"_3D"] = score_3d

                elif self.net.__class__.__name__ in ["DoseNet3D", "SegNet3D"]:
                    for bn in range(outputs.shape[0]):
                        for slice in range(outputs.shape[2]):
                            if self.net.__class__.__name__ == "SegNet3D":
                                score = metric(outputs[bn, :, slice, None].permute(1,0,2,3), labels[bn, slice, None])
                            else:
                                score = metric(outputs[bn, slice, None], labels[bn, slice, None])
                            try:
                                self.test_scores[self.metric_names[i]+"_2D"] = torch.cat((self.test_scores[self.metric_names[i]+"_2D"], score), dim=1)
                            except:
                                self.test_scores[self.metric_names[i]+"_2D"] = score
              

                    if self.net.__class__.__name__ == "SegNet3D":     
                        score_3d = metric(outputs, labels)
                    else:     
                        score_3d = metric(outputs, labels) 
                    try:
                        self.test_scores[self.metric_names[i]+"_3D"] = torch.cat((self.test_scores[self.metric_names[i]+"_3D"], score_3d), dim=1)
                    except:
                        self.test_scores[self.metric_names[i]+"_3D"] = score_3d

        for key in self.test_scores.keys():
            if key != self.loss_name:
                means = []
                stds = []
                maskisnan = torch.isnan(self.test_scores[key][0])
                maskisinf = torch.isinf(self.test_scores[key][0])
                means.append(torch.mean(self.test_scores[key][0][~(maskisnan | maskisinf)]))
                stds.append(torch.std(self.test_scores[key][0][~(maskisnan | maskisinf)]))
                self.test_scores[key]=torch.stack((torch.tensor(means), torch.tensor(stds)), dim=1)
            else:
                self.test_scores[key] = self.test_scores[key]/n

        for metric in self.test_scores:
            if metric!=self.loss_name:
                np.savetxt(self.CHECKPOINTS_DIR / f"{metric}_test_scores.csv", self.test_scores[metric][:, 0, 0], delimiter=",")
                
        with open(self.CHECKPOINTS_DIR / "test_results.txt", "w") as file:
            file.write(f"Mean valid {self.loss_name} = {self.minimum_valid_loss}\n")
            for score in self.test_scores:
                if score != self.loss_name:
                    for c in range(self.test_scores[score].shape[0]):
                        if self.net.__class__.__name__ in ["SegNet2D", "SegNet3D"]:
                            if c==0:
                                file.write(f"Mean test {score} Background = {self.test_scores[score][c].numpy()}\n")
                            elif c > len(self.label_info['label_info']):
                                file.write(f"Mean test {score} Average = {self.test_scores[score][c].numpy()}\n")
                            else:
                                file.write(f"Mean test {score} {self.label_info['label_info'][c][0]} = {self.test_scores[score][c].numpy()}\n")
                        else:
                            file.write(f"Mean test {score} = {self.test_scores[score]}\n")
                else:
                    file.write(f"Mean test {score} = {self.test_scores[score]}\n")
                file.write("\n")
    
        return self.test_scores[self.loss_name] 
  
