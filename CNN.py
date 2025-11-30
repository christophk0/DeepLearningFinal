import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50
import time


class CNN(nn.Module):
    def __init__(self, config, device, num_classes=10):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.config = config
        
        # Select ResNet architecture
        resnet_type = config.get('architecture', 'resnet18')
        if resnet_type == 'resnet18':
            self.resnet = resnet18(
                weights=models.ResNet18_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
        elif resnet_type == 'resnet34':
            self.resnet = resnet34(
                weights=models.ResNet34_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
        elif resnet_type == 'resnet50':
            self.resnet = resnet50(
                weights=models.ResNet50_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {resnet_type}")
        drop = config['num_layers_to_drop']
        if drop > 0:
            self.resnet.layer4 = nn.Identity()
            if drop > 1:
                self.resnet.layer3 = nn.Identity()
                if drop > 2:
                    self.resnet.layer2 = nn.Identity()
                    if drop > 3:
                        self.resnet.layer1 = nn.Identity()
                        if drop > 4:
                            self.resnet.layer0 = nn.Identity()
        self.resnet.fc = nn.Identity().to(self.device)
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        print(self.resnet)

        # Get the feature size from ResNet
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.resnet(dummy_input)
            feature_size = features.shape[1]
        
        self.final_layer = nn.Linear(feature_size, self.num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer with optional weight decay and scheduler
        optimizer_params = self.final_layer.parameters() if self.freeze_pretrained_layers else self.parameters()
        self.optimizer = optim.Adam(
            optimizer_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Optional learning rate scheduler
        if config.get('use_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.get('scheduler_step_size', 30),
                gamma=config.get('scheduler_gamma', 0.1)
            )
        else:
            self.scheduler = None

    def forward(self, x):
        x = self.resnet(x)
        return self.final_layer(x)

    def training_step(self, batch):
        self.train()
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        
        start_time = time.time()
        self.optimizer.zero_grad()
        output = self.forward(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        # Optional gradient clipping
        if self.config.get('grad_clip', None):
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config['grad_clip'])
        
        self.optimizer.step()
        
        # Step scheduler if available
        if self.scheduler:
            self.scheduler.step()
        
        training_time = time.time() - start_time
        
        # Return loss and additional metrics
        with torch.no_grad():
            pred = output.argmax(dim=1)
            accuracy = (pred == target).float().mean().item()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'training_time': training_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def test_step(self, batch):
        self.eval()
        with torch.no_grad():
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            
            start_time = time.time()
            output = self.forward(data)
            inference_time = time.time() - start_time
            
            test_loss = self.criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            
            return {
                'loss': test_loss,
                'correct': correct,
                'total': target.size(0),
                'predictions': pred.cpu().numpy(),
                'targets': target.cpu().numpy(),
                'inference_time': inference_time
            }
    
    def get_model_info(self):
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN',
            'architecture': self.config.get('architecture', 'resnet18'),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pretrained': self.config['pretrained'],
            'frozen_layers': self.freeze_pretrained_layers,
            'layers_dropped': self.config['num_layers_to_drop'],
            'num_classes': self.num_classes
        }
