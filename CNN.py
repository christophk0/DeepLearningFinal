import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.optim.lr_scheduler import CosineAnnealingLR


class CNN(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        if config['model'] == 18:
            self.resnet = resnet18(
                weights=models.ResNet18_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        elif config['model'] == 34:
            self.resnet = resnet34(
                weights=models.ResNet34_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        elif config['model'] == 50:
            self.resnet = resnet50(
                weights=models.ResNet50_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        elif config['model'] == 101:
            self.resnet = resnet101(
                weights=models.ResNet101_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        elif config['model'] == 152:
            self.resnet = resnet152(
                weights=models.ResNet152_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        else:
            exit("Unsupported Resnet model type. Supported models are 18, 34, 50, 101, 152")
        for i in range(min(config['num_layers_to_drop'], 4)):
            setattr(self.resnet, 'layer{}'.format(4 - i), nn.Identity())
        self.resnet.fc = nn.Identity().to(self.device)
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        print(self.resnet)
        total_params = sum(param.numel() for param in self.parameters())
        print('Total number of parameters (excluding final linear layer): {}'.format(total_params))

        self.final_layer = nn.LazyLinear(10).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.final_layer.parameters() if self.freeze_pretrained_layers else self.parameters(),
            lr=config['learning_rate'])
        self.scheduler = None
        if config['enable_cosine_scheduler']:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['cosine_scheduler_max'])

    def forward(self, x):
        x = self.resnet(x)
        return self.final_layer(x)

    def training_step(self, batch):
        self.train()
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.forward(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_step(self, batch):
        self.eval()
        with torch.no_grad():
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            pred = self.forward(data)
            test_loss = self.criterion(pred, target).item()
            correct = (pred.argmax(1) == target).type(torch.float).sum().item()
            return test_loss, correct
