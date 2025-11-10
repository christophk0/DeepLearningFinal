import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet18


class CNN(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.model = resnet18(
            weights=models.ResNet18_Weights.DEFAULT if config['pretrained'] else None).to(
            self.device)
        drop = config['num_layers_to_drop']
        if drop > 0:
            self.model.layer4 = nn.Identity()
            if drop > 1:
                self.model.layer3 = nn.Identity()
                if drop > 2:
                    self.model.layer2 = nn.Identity()
                    if drop > 3:
                        self.model.layer1 = nn.Identity()
                        if drop > 4:
                            self.model.layer0 = nn.Identity()


        self.model.fc = nn.LazyLinear(10).to(self.device)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, epoch, batch_idx, print_debug: bool = False):
        self.model.train()
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        if print_debug:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    def test_step(self, batch):
        self.model.eval()
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        pred = self.model(data)
        test_loss = self.criterion(pred, target).item()
        correct = (pred.argmax(1) == target).type(torch.float).sum().item()
        return test_loss, correct
