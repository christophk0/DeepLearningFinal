import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet18


class CNN(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        self.resnet = resnet18(
            weights=models.ResNet18_Weights.DEFAULT if config['pretrained'] else None).to(
            self.device)
        for i in range(min(config['num_layers_to_drop'], 4)):
            setattr(self.resnet, 'layer{}'.format(4 - i), nn.Identity())
        self.resnet.fc = nn.Identity().to(self.device)
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        print(self.resnet)

        self.final_layer = nn.LazyLinear(10).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.final_layer.parameters() if self.freeze_pretrained_layers else self.parameters(),
            lr=config['learning_rate'])

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
