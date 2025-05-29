import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import gc

from scipy.stats import wilcoxon

from src.datasets import CUBDataset, VOCSegDataset, DIV2KDataset
from src.models import Classifier, Segmentator, Transformer
from src.energy import FieldOptimizer2D, FieldOptimizerShifts #, FieldOptimizer3D



### Segmentation on VOCSeg dataset

class VOCSegInference:
    def __init__(self, 
                 device, 
                 coupling_constant=1e6, 
                 alpha=1.0,
                 beta=1.0,
                 v=0.01,
                 coarse_factor=1 
                 ):
        
        self.device = device
        self.coupling_constant=coupling_constant
        self.alpha=alpha
        self.beta=beta
        self.v=v
        self.coarse_factor=coarse_factor      

    def run(self, train_epochs=5, eps=500, detection_steps=10, max_samples=None, thresh=1e-3):
        train_ds = VOCSegDataset('./data', image_set='train')
        val_ds = VOCSegDataset('./data', image_set='val')
    
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    
        segmentator = Segmentator(self.device, num_classes=21)
        

        if train_epochs:
            segmentator.train(train_loader, val_loader, num_epochs=1)
            torch.save(segmentator.model.state_dict(),"checkpoints/deeplabv3.pth")

        segmentator.model.load_state_dict(torch.load('checkpoints/deeplabv3.pth', map_location=self.device))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

        lie = FieldOptimizer2D()
        
        # eps = 500.0  # Flow time
        # detection_steps = 10

        
        segmentator.model.eval()

        losses_in = []
        losses_new = []    
        in_correct = 0
        new_correct = 0
        in_metrics = []
        new_metrics = []
        total = 0

        total_detections = 0
        
        for idx, (imgs, labels) in enumerate(val_loader):
        
            if max_samples is not None and idx >= max_samples:
                break
                    
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            channels = imgs.size(1)

            total += labels.size(0)
        
            with torch.no_grad():
                output_before, predicted, in_loss, is_correct = segmentator.forward(imgs, labels)
            
            losses_in.append(in_loss.detach().cpu())
            in_correct += is_correct 
            
            ious, in_metric = segmentator.compute_iou(predicted,labels)
            # dices, in_metric = segmentator.compute_dice(predicted,labels)
            in_metrics.append(in_metric)
        
            # Compute phi field
            img_prime_list = []
            for c in range(channels):
                single_channel_img = imgs[:,c,:,:].unsqueeze(1)
                img_prime_c, phi_sparse, phi = lie.compute_energy(single_channel_img,
                                                            coupling_constant=self.coupling_constant,
                                                            alpha=self.alpha,
                                                            beta=self.beta,
                                                            v=self.v,
                                                            coarse_factor=self.coarse_factor)        
                img_prime_list.append(img_prime_c.squeeze(1))
            
            img_prime = torch.stack(img_prime_list,dim=0)
            img_prime = img_prime.permute(1,0,2,3).to(self.device)
        
            
            # Calculate the sign of target metric along two rays +/- h with small flow time = t
            for step in range(detection_steps):
                t = eps*(detection_steps-step)/(detection_steps)
                h = t*(img_prime-imgs)
                # h = t*torch.rand_like(imgs)*0.1
                with torch.no_grad():
                    # Ray "+h"
                    output_plus, plus_predicted, loss_plus, plus_correct = segmentator.forward(imgs+h, labels)
                    _, plus_true_scores = segmentator.compute_iou(plus_predicted,labels)
                    # _, plus_true_scores = segmentator.compute_dice(plus_predicted,labels)
        
                    # Ray "-h"
                    output_minus, minus_predicted, loss_minus, minus_correct = segmentator.forward(imgs-h, labels)
                    _, minus_true_scores = segmentator.compute_iou(minus_predicted,labels)
                    # _, minus_true_scores = segmentator.compute_dice(plus_predicted,labels)
                
                if plus_true_scores > in_metric + thresh:
                    dot = 1
                    break
                elif minus_true_scores > in_metric + thresh:
                    dot = -1
                    break
                else:
                    dot = 0

            total_detections += abs(dot)
        
            # Apply warp along the chosen ray with flow time = eps
            dynamic_eps = dot*t
            print(f"Chosen flow time: {dynamic_eps}")
            h = dynamic_eps*(img_prime-imgs)

            if dot != 0:
                with torch.no_grad():
                    output_after, new_predicted, loss_after, correct_after = segmentator.forward(imgs+h, labels) 
            
                losses_new.append(loss_after.detach().cpu())
                new_correct += correct_after    
                _, new_metric = segmentator.compute_iou(new_predicted,labels)
                # dices, in_metric = segmentator.compute_dice(new_predicted,labels)
            else:
                losses_new.append(in_loss.detach().cpu())
                new_correct += is_correct 
                new_metric = in_metric
                
                
            new_metrics.append(new_metric)
        
            if idx%1 == 0:
                # print(f"OLD: {predicted.item()}")
                # print(f"NEW: {new_predicted.item()}")
                # print(f"TRU: {labels[0]}")
                print(f"IoU diff: {100*(new_metric - in_metric)/(in_metric+1e-7):.2f}%\n")
        
            del imgs, img_prime, single_channel_img, phi_sparse, phi, img_prime_c
            gc.collect()
            torch.cuda.empty_cache()


        # Statistics

        print(f"total_detections: {total_detections}/{total} [{100*total_detections/total:.2f}%]")
        
        in_acc = 100.*in_correct/total
        new_acc = 100.*new_correct/total
        
        print(f"Accuracy: {in_acc} --> {new_acc}")
        print(f"Loss: {np.mean(losses_in):.6f} --> {np.mean(losses_new):.6f}")
        print(f"IoU: {np.mean(in_metrics):.6f} --> {np.mean(new_metrics):.6f}")   
 
        stat, p_value = wilcoxon(losses_in, losses_new)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}")

        stat, p_value = wilcoxon(in_metrics, new_metrics)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}") 









        

### Classification on Birds CUB-200 dataset

class CUBInference:
    def __init__(self, 
                 device, 
                 coupling_constant=1e6, 
                 alpha=1.0,
                 beta=1.0,
                 v=0.01,
                 coarse_factor=1 
                 ):
        
        self.device = device
        self.coupling_constant=coupling_constant
        self.alpha=alpha
        self.beta=beta
        self.v=v
        self.coarse_factor=coarse_factor      

    def run(self, train_epochs=5, eps=500, detection_steps=10, max_samples=None):
        train_ds = CUBDataset(root_dir='datasets/birds/CUB_200_2011', split="train")
        val_ds = CUBDataset(root_dir='datasets/birds/CUB_200_2011', split="val")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)

        classifier = Classifier(mode='efficientnet', num_classes=200, device=self.device)

        if train_epochs:
            classifier.train(train_loader, val_loader, num_epochs=train_epochs)
            torch.save(classifier.model.state_dict(),"checkpoints/efficientnet_birds.pth")

        classifier.model.load_state_dict(torch.load('checkpoints/efficientnet_birds.pth', map_location=self.device))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

        lie = FieldOptimizer2D()
        
        # eps = 500.0  # Flow time
        # detection_steps = 10

        
        classifier.model.eval()

        losses_in = []
        losses_new = []    
        in_correct = 0
        new_correct = 0
        in_metrics = []
        new_metrics = []
        total = 0

        total_detections = 0
        
        for idx, (imgs, labels) in enumerate(val_loader):
        
            if max_samples is not None and idx >= max_samples:
                break
                    
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            channels = imgs.size(1)

            total += labels.size(0)
        
            with torch.no_grad():
                output_before, predicted, in_loss = classifier.forward(imgs, labels)  # [B, D, C]
            
            losses_in.append(in_loss.detach().cpu())
            in_correct += predicted.eq(labels).sum().item()    
        
            in_metric = classifier.compute_ece(output_before,labels)
            in_metrics.append(in_metric)
        
            # Compute phi field
            img_prime_list = []
            for c in range(channels):
                single_channel_img = imgs[:,c,:,:].unsqueeze(1)
                img_prime_c, phi_sparse, phi = lie.compute_energy(single_channel_img,
                                                            coupling_constant=self.coupling_constant,
                                                            alpha=self.alpha,
                                                            beta=self.beta,
                                                            v=self.v,
                                                            coarse_factor=self.coarse_factor)        
                img_prime_list.append(img_prime_c.squeeze(1))
            
            img_prime = torch.stack(img_prime_list,dim=0)
            img_prime = img_prime.permute(1,0,2,3).to(self.device)
        
            thresh = 1e-3
            # Calculate the sign of target metric along two rays +/- h with small flow time = t
            for step in range(detection_steps):
                t = eps*(detection_steps-step)/(detection_steps)
                h = t*(img_prime-imgs)
                # h = t*torch.rand_like(imgs)*0.1
                with torch.no_grad():
                    # Ray "+h"
                    output_plus, plus_predicted, loss_plus = classifier.forward(imgs+h, labels)
                    plus_true_scores = classifier.compute_ece(output_plus,labels)
                    plus_correct = plus_predicted.eq(labels).sum().item()   
        
                    # Ray "-h"
                    output_minus, minus_predicted, loss_minus = classifier.forward(imgs-h, labels)
                    minus_true_scores = classifier.compute_ece(output_minus,labels)
                    minus_correct = minus_predicted.eq(labels).sum().item()
                
                if plus_correct > predicted.eq(labels).sum().item(): #plus_true_scores < in_metric - thresh:
                    dot = 1
                    break
                elif minus_correct > predicted.eq(labels).sum().item(): #minus_true_scores < in_metric - thresh:
                    dot = -1
                    break
                else:
                    dot = 0

            total_detections += abs(dot)
        
            # Apply warp along the chosen ray with flow time = eps
            dynamic_eps = dot*t
            print(f"Chosen flow time: {dynamic_eps}")
            h = dynamic_eps*(img_prime-imgs)

            if dot != 0:
                with torch.no_grad():
                    output_after, new_predicted, loss_after = classifier.forward(imgs+h, labels) 
            
                losses_new.append(loss_after.detach().cpu())
                new_correct += new_predicted.eq(labels).sum().item()    
                new_metric = classifier.compute_ece(output_after,labels)
            else:
                losses_new.append(in_loss.detach().cpu())
                new_correct += predicted.eq(labels).sum().item()  
                new_metric = in_metric
                
                
            new_metrics.append(new_metric)
        
            if idx%1 == 0:
                # print(f"OLD: {predicted.item()}")
                # print(f"NEW: {new_predicted.item()}")
                # print(f"TRU: {labels[0]}")
                print(f"ECE: {100*(new_metric - in_metric)/(in_metric+1e-7):.2f}%\n")
        
            del imgs, img_prime, single_channel_img, phi_sparse, phi, img_prime_c
            gc.collect()
            torch.cuda.empty_cache()


        # Statistics

        print(f"total_detections: {total_detections}/{total} [{100*total_detections/total:.2f}%]")
        
        in_acc = 100.*in_correct/total
        new_acc = 100.*new_correct/total
        
        print(f"Accuracy: {in_acc} --> {new_acc}")
        print(f"Loss: {np.mean(losses_in):.6f} --> {np.mean(losses_new):.6f}")
        print(f"ECE: {np.mean(in_metrics):.6f} --> {np.mean(new_metrics):.6f}")   
 
        stat, p_value = wilcoxon(losses_in, losses_new)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}")

        stat, p_value = wilcoxon(in_metrics, new_metrics)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}")         



### Perceptual loss on DIV2K dataset

class ViTInference:
    def __init__(self, 
                 device, 
                 coupling_constant=1e6, 
                 alpha=1.0,
                 beta=1.0,
                 v=0.01,
                 coarse_factor=1,
                 num_steps=50
                 ):
        
        self.device = device
        self.coupling_constant=coupling_constant
        self.alpha=alpha
        self.beta=beta
        self.v=v
        self.coarse_factor=coarse_factor  

        self.num_steps = num_steps

    
    
    def run(self, train_epochs=None, eps=500, detection_steps=10, batch_size=1, max_samples=None, thresh=1e-3):

        val_ds = DIV2KDataset(root_dir='datasets/div2k', scale=4, split='valid')  
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)


        transformer = Transformer(model_name='vit_base_patch8_224', pretrained=True, output='patch', device=self.device)
        transformer.model.eval()

        B, C, H, W = batch_size, 768, 28, 28

        lie = FieldOptimizerShifts(B, C, H, W, 
                                   coupling_constant=1.0, alpha=0.1, beta=0.1, v=0.1, 
                                   lr=0.05, coarse_factor=1, num_steps=self.num_steps, 
                                   lie_dim=2)

        losses_in = []
        losses_new = []    
        in_correct = 0
        new_correct = 0
        in_metrics = []
        new_metrics = []
        total = 0

        total_detections = 0
        
        for idx, (lr_img, hr_img) in enumerate(val_loader):
        
            if max_samples is not None and idx >= max_samples:
                break
                    
            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)
            total += hr_img.size(0)

            with torch.no_grad():
                in_features = transformer.forward(lr_img)
                ref_features = transformer.forward(hr_img)
                in_loss = transformer.perceptual_patchwise_loss(in_features, ref_features).sum()
                losses_in.append(in_loss.detach().cpu())

                # Reshape: (B, N, C) → (B, H, W, C) → (B, C, H, W)
                in_features = in_features.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, D, H, W]
                ref_features = ref_features.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, D, H, W]

                print(f"in_features: {in_features.shape}, ref_features: {ref_features.shape}")



            features_prime, phi_sparse, phi = lie.compute_energy(in_features)   
                

            # Calculate the sign of target metric along two rays +/- h with small flow time = t
            for step in range(detection_steps):
                t = eps*(detection_steps-step)/(detection_steps)
                h = t*(features_prime-in_features)
                # h = t*torch.rand_like(imgs)*0.1

                print(f"S: {in_features.abs().sum()}, S':{features_prime.abs().sum()}, h: {h.abs().sum()}")

                with torch.no_grad():
                    # Ray "+h"
                    loss_plus = transformer.perceptual_patchwise_loss(in_features+h, ref_features).sum()
                    # Ray "-h"
                    loss_minus = transformer.perceptual_patchwise_loss(in_features-h, ref_features).sum()

                    print(f"Loss: {in_loss.item()}, +h: {loss_plus.item()}, -h: {loss_minus.item()}")
                
                if loss_plus < in_loss - thresh:
                    dot = 1
                    break
                elif loss_minus < in_loss - thresh:
                    dot = -1
                    break
                else:
                    dot = 0

            total_detections += abs(dot)
        
            # Apply warp along the chosen ray with flow time = eps
            dynamic_eps = dot*t
            print(f"Chosen flow time: {dynamic_eps}")
            h = dynamic_eps*(features_prime-in_features)

            if dot != 0:
                with torch.no_grad():
                    loss_after = transformer.perceptual_patchwise_loss(in_features+h, ref_features).sum()
            
                losses_new.append(loss_after.detach().cpu())

            else:
                loss_after = in_loss
                losses_new.append(loss_after.detach().cpu())

            if idx%1 == 0:
                print(f"Loss diff: {100*(loss_after - in_loss)/(in_loss+1e-7):.2f}%\n")
        
            del lr_img, hr_img, in_features, ref_features, features_prime, phi_sparse, phi
            gc.collect()
            torch.cuda.empty_cache()


        # Statistics

        print(f"total_detections: {total_detections}/{total} [{100*total_detections/total:.2f}%]")
        
        in_acc = 100.*in_correct/total
        new_acc = 100.*new_correct/total
        
        print(f"Accuracy: {in_acc} --> {new_acc}")
        print(f"Loss: {np.mean(losses_in):.6f} --> {np.mean(losses_new):.6f}")
        print(f"ECE: {np.mean(in_metrics):.6f} --> {np.mean(new_metrics):.6f}")   
 
        stat, p_value = wilcoxon(losses_in, losses_new)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}")

        stat, p_value = wilcoxon(in_metrics, new_metrics)
        print(f"Wilcoxon statistic: {stat}")
        print(f"p-value: {p_value:.4e}")  
        

        