import timm
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm

import torchvision.models.segmentation as models
import torch.nn.functional as F



class Segmentator:
    def __init__(self, device='cpu', num_classes=21):
        self.device = device
        self.model = self.get_deeplabv3(num_classes)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC uses 255 for void


    def get_deeplabv3(self, num_classes=21):
        model = models.deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model

    def forward(self, images, masks):
        outputs = self.model(images)['out']  # [B, C, H, W]
        preds = outputs.argmax(dim=1)   # [B, H, W]
        valid = masks != 255
        is_correct = (preds[valid] == masks[valid]).sum().item()
        loss = self.criterion(outputs, masks)

        return outputs, preds, loss, is_correct

    def train(self, train_loader, val_loader, num_epochs=10):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            total = correct = 0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(self.device)
                masks = masks.squeeze(1).to(self.device)  # [B, H, W] for loss

                optimizer.zero_grad()
                outputs = self.model(images)['out']  # [B, C, H, W]
                preds = outputs.argmax(dim=1)   # [B, H, W]
                valid = masks != 255
                correct += (preds[valid] == masks[valid]).sum().item()
                total += valid.sum().item()

                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            avg_loss = running_loss / len(train_loader.dataset)
            acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Accuracy: {acc:.2f}")
            self.val(val_loader)

    def val(self, val_loader):
        self.model.eval()
        total = correct = 0

        for images, masks in val_loader:
            images = images.to(self.device)
            masks = masks.squeeze(1).to(self.device)

            outputs = self.model(images)['out']  # [B, C, H, W]
            preds = outputs.argmax(dim=1)   # [B, H, W]

            valid = masks != 255
            correct += (preds[valid] == masks[valid]).sum().item()
            total += valid.sum().item()

        acc = 100.0 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

    import torch

    def compute_iou(self, preds, targets, num_classes=21, ignore_index=255):
        """
        preds: tensor of shape [B, H, W] (predicted class indices)
        targets: tensor of shape [B, H, W] (ground-truth class indices)
        Returns:
            per-class IoU, and mean IoU
        """
        ious = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
    
            pred_inds = (preds == cls)
            target_inds = (targets == cls)
            mask = (targets != ignore_index)
    
            intersection = (pred_inds & target_inds & mask).sum().item()
            union = (pred_inds | target_inds) & mask
            union = union.sum().item()
    
            if union == 0:
                ious.append(float('nan'))  # ignore this class
            else:
                ious.append(intersection / union)
    
        miou = torch.tensor([i for i in ious if not torch.isnan(torch.tensor(i))]).mean().item()
        return ious, miou

    def compute_dice(self, preds, targets, num_classes=21, ignore_index=255):
        """
        preds: tensor of shape [B, H, W]
        targets: tensor of shape [B, H, W]
        Returns:
            per-class Dice score, and mean Dice
        """
        dice_scores = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
    
            pred_inds = (preds == cls)
            target_inds = (targets == cls)
            mask = (targets != ignore_index)
    
            intersection = (pred_inds & target_inds & mask).sum().item()
            pred_area = (pred_inds & mask).sum().item()
            target_area = (target_inds & mask).sum().item()
    
            denom = pred_area + target_area
            if denom == 0:
                dice_scores.append(float('nan'))
            else:
                dice_scores.append(2 * intersection / denom)
    
        mdice = torch.tensor([d for d in dice_scores if not torch.isnan(torch.tensor(d))]).mean().item()
        return dice_scores, mdice



class Classifier:
    def __init__(self, mode='efficientnet', num_classes=200, device='cpu'):
        self.device = device
        if mode == 'efficientnet':
            self.model = self.get_efficientnet_model(num_classes=200)
            print(f"EfficientNet model initialized for {num_classes} classes")

        self.criterion = nn.CrossEntropyLoss()
        self.model.to(device)
        

    def get_efficientnet_model(self, num_classes=200):
        model = timm.create_model('efficientnet_b0', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    def forward(self, imgs, labels):
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            _, predicted = outputs.max(1)

        return outputs, predicted, loss
        

    def train(self, train_loader, val_loader, num_epochs=10):

        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)        


        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                if idx%10 == 0:
                    print(f"Epoch {epoch}, step {idx}/{len(train_loader)}")

            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total

            val_loss, val_acc = self.val(val_loader)

            print(f"\nEpoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
            print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.2f}%\n")

            # scheduler.step()        

    def val(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            val_loss = val_loss / total
            val_acc = 100. * correct / total       

        return val_loss, val_acc


    def compute_ece(self, logits, labels, n_bins=15):
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)
    
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            bin_size = mask.sum().item()
            if bin_size > 0:
                acc = accuracies[mask].float().mean().item()
                conf = confidences[mask].mean().item()
                ece += (bin_size / len(logits)) * abs(acc - conf)
        return ece        
    



class Transformer(nn.Module):
    def __init__(self, model_name='vit_base_patch8_224', pretrained=True, output='patch', device='cpu'):
        """
        Args:
            model_name: any ViT model from timm (default: ViT-B/8)
            pretrained: load pretrained weights (on ImageNet-1K)
            output: 'cls', 'mean', 'tokens', or 'patch'
        """
        super().__init__()
        self.device = device
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.output_type = output

        # Strip classification head
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity()
        elif hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()

        self.model.to(self.device)

    def forward(self, x):

        try:
            tokens = self.model.forward_features(x)  # [B, 1+N, D]
        except Exception as e:
            print(f"Unable to extract features")
            print(f"{e}")

        # if self.output_type == 'cls':
        #     return tokens[:, 0]  # [B, D]
        # elif self.output_type == 'mean':
        #     return tokens[:, 1:].mean(dim=1)  # [B, D]
        # elif self.output_type == 'tokens':
        #     return tokens  # [B, 1+N, D]
        # elif self.output_type == 'patch':
        #     return tokens[:, 1:]  # [B, N, D] — all spatial tokens, no CLS
        # else:
        #     raise ValueError(f"Invalid output_type: {self.output_type}")

