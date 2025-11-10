import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import vit_b_16


class VisionTransformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.model = vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT if config['pretrained'] else None).to(
            self.device)
        self.model.heads.head = nn.LazyLinear(10).to(self.device)
        self.model.encoder.layers = self.model.encoder.layers[
            :-config['num_encoder_layers_to_drop']]
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
