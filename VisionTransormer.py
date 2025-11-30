import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import vit_b_16, vit_b_32, vit_l_16
from local_attention import LocalVisionTransformer
import time


class VisionTransformer(nn.Module):
    def __init__(self, config, device, num_classes=10):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.config = config
        self.use_local_attention = config.get('use_local_attention', False)
        
        if self.use_local_attention:
            # Use custom local attention ViT
            self.vit = LocalVisionTransformer(
                img_size=224,
                patch_size=16,
                num_classes=num_classes,
                embed_dim=config.get('embed_dim', 768),
                depth=config.get('depth', 12),
                num_heads=config.get('num_heads', 12),
                window_size=config.get('local_window_size', 7),
                dropout=config.get('dropout', 0.0)
            ).to(self.device)
            self.final_layer = None  # Local ViT has its own head
        else:
            # Use standard pretrained ViT
            vit_type = config.get('architecture', 'vit_b_16')
            if vit_type == 'vit_b_16':
                self.vit = vit_b_16(
                    weights=models.ViT_B_16_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
            elif vit_type == 'vit_b_32':
                self.vit = vit_b_32(
                    weights=models.ViT_B_32_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
            elif vit_type == 'vit_l_16':
                self.vit = vit_l_16(
                    weights=models.ViT_L_16_Weights.DEFAULT if config['pretrained'] else None).to(self.device)
            else:
                raise ValueError(f"Unsupported ViT architecture: {vit_type}")
            
            self.vit.heads.head = nn.Identity().to(self.device)
            
            # Get the feature size from ViT
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                features = self.vit(dummy_input)
                feature_size = features.shape[1]
            
            self.final_layer = nn.Linear(feature_size, self.num_classes).to(self.device)
            
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers and not self.use_local_attention:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Drop encoder layers if specified
        if config['num_encoder_layers_to_drop'] > 0 and not self.use_local_attention:
            self.vit.encoder.layers = self.vit.encoder.layers[
                :-config['num_encoder_layers_to_drop']]
        
        print(self.vit)


        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if self.use_local_attention:
            optimizer_params = self.parameters()
        else:
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
        if self.use_local_attention:
            return self.vit(x)
        else:
            x = self.vit(x)
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
        
        info = {
            'model_type': 'ViT',
            'architecture': 'local_attention' if self.use_local_attention else self.config.get('architecture', 'vit_b_16'),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pretrained': self.config['pretrained'] if not self.use_local_attention else False,
            'frozen_layers': self.freeze_pretrained_layers,
            'encoder_layers_dropped': self.config['num_encoder_layers_to_drop'],
            'num_classes': self.num_classes,
            'use_local_attention': self.use_local_attention
        }
        
        if self.use_local_attention:
            info['local_window_size'] = self.config.get('local_window_size', 7)
            info['embed_dim'] = self.config.get('embed_dim', 768)
            info['depth'] = self.config.get('depth', 12)
            info['num_heads'] = self.config.get('num_heads', 12)
        
        return info
