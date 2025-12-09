import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import vit_b_16, vit_l_16
from types import MethodType


def modified_block_forward(self, input: torch.Tensor):
    # Standard torchvision checks
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

    x = self.ln_1(input)
    x, _ = self.self_attention(x, x, x, need_weights=False)
    x = self.dropout(x)

    y = self.ln_2(x)
    y = self.mlp(y)
    return y

class VisionTransformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        if config['model'] == 'base':
            self.vit = vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        elif config['model'] == 'large':
            self.vit = vit_l_16(
                weights=models.ViT_L_16_Weights.DEFAULT if config['pretrained'] else None).to(
                self.device)
        self.vit.heads.head = nn.Identity().to(self.device)
        self.freeze_pretrained_layers = config['freeze_pretrained_layers']
        if self.freeze_pretrained_layers:
            for param in self.vit.parameters():
                param.requires_grad = False
        if config['num_encoder_layers_to_drop'] > 0:
            self.vit.encoder.layers = self.vit.encoder.layers[
                :-config['num_encoder_layers_to_drop']]
        if config.get('drop_residual_connections', 0) > 0:
            target_block = self.vit.encoder.layers[config.get('drop_residual_connections', 0)]
            print(f"Dropping residual connections in block {config.get('drop_residual_connections', 0)}")
            target_block.forward = MethodType(modified_block_forward, target_block)
        print(self.vit)

        total_params = sum(param.numel() for param in self.parameters())
        print('Total number of parameters (excluding final linear layer): {}'.format(total_params))

        self.final_layer = nn.LazyLinear(10).to(self.device)
        
        if config.get('path_to_weights', '') != '':
            print(f"Loading weights from {config.get('path_to_weights')}")
            self.load_state_dict(torch.load(config.get('path_to_weights')))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.final_layer.parameters() if self.freeze_pretrained_layers else self.parameters(),
            lr=config['learning_rate'])
        self.scheduler = None
        if config['enable_cosine_scheduler']:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['cosine_scheduler_max'])

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
