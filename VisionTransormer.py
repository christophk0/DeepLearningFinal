import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import vit_b_16


class VisionTransformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        self.vit = vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT if config['pretrained'] else None).to(
            self.device)
        self.vit.heads.head = nn.Identity().to(self.device)
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers:
            for param in self.vit.parameters():
                param.requires_grad = False
        if config['num_encoder_layers_to_drop'] > 0:
            self.vit.encoder.layers = self.vit.encoder.layers[
                :-config['num_encoder_layers_to_drop']]
        print(self.vit)

        self.final_layer = nn.LazyLinear(10).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.final_layer.parameters() if self.freeze_pretrained_layers else self.parameters(),
            lr=config['learning_rate'])

    def forward(self, x):
        x = self.vit(x)
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
