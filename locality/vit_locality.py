import argparse

import torch
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import vit_b_16
from torchvision.models import vit_l_16
from torchvision.models import vit_h_14
from torchvision import datasets, transforms
import numpy as np
import os
import distinctipy


def get_attention_forward(block_idx, attention_dict):
    # Below is code from vision_transformer implementation. We need to modify it to store attention data.
    # def forward(self, input: torch.Tensor):
    #     torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    #     x = self.ln_1(input)
    #     x, _ = self.self_attention(x, x, x, need_weights=False)
    #     x = self.dropout(x)
    #     x = x + input
    #
    #     y = self.ln_2(x)
    #     y = self.mlp(y)
    #     return x + y
    def forward_wrapper(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, attn_weights = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)

        # Modification: store weights.
        attention_dict[block_idx] = attn_weights.detach()

        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    return forward_wrapper


def compute_distance_matrix(patch_size, num_patches_per_side):
    grid_coordinates = []
    for r in range(num_patches_per_side):
        for c in range(num_patches_per_side):
            grid_coordinates.append(np.array([r, c]))

    num_patches = num_patches_per_side * num_patches_per_side
    dist_matrix = np.zeros((num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            dist_matrix[i, j] = patch_size * np.linalg.norm(grid_coordinates[i] - grid_coordinates[j])

    return torch.tensor(dist_matrix, dtype=torch.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    vit_weights = None
    if args.weights is not None:
        weights = torch.load(args.weights, map_location=torch.device('cpu'))
        # We remove the head in our implementation, so replace them with 0s. They aren't relevant for attention
        # calculations.
        vit_weights = {'heads.head.weight': torch.zeros(1000, 768), 'heads.head.bias': torch.zeros(1000)}
        for key, value in weights.items():
            if key.startswith('vit.'):
                vit_weights[key[4:]] = value

    if args.model == 'base':
        model = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        patch_size = 16
        image_size = 224
        title = 'VIT-B/16'
    elif args.model == 'large':
        model = vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        patch_size = 16
        image_size = 224
        title = 'VIT-L/16'
    elif args.model == 'huge':
        model = vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
        patch_size = 14
        image_size = 518
        title = 'VIT-H/14'
    else:
        raise NotImplementedError('Unsupported model type')

    if vit_weights is not None:
        title += ' (trained from scratch)'
        model.load_state_dict(vit_weights)

    model.eval()

    # Store attention during model forward.
    attention_dict = {}

    for i, layer in enumerate(model.encoder.layers):
        # Replace with layer with same output that stores attention.
        # Must bind function to layer instance explicitly with __get__
        layer.forward = get_attention_forward(i, attention_dict).__get__(layer, type(layer))

    # Precompute distances between patches.
    dist_matrix = compute_distance_matrix(patch_size=patch_size, num_patches_per_side=int(image_size / patch_size))

    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    images = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    img = images[0][0].unsqueeze(0)

    # Run modified model to update attention_dict.
    with torch.no_grad():
        model(img)

    mean_distances = []

    for layer_idx in sorted(attention_dict.keys()):
        # Retrieve saved attention.
        attn = attention_dict[layer_idx]

        # Remove Batch dim and CLS token
        patch_attn = attn[0, :, 1:, 1:]

        # Normalize
        patch_attn = patch_attn / patch_attn.sum(dim=-1, keepdim=True)

        # Use precomputed distance to weights attentions for each head and patch.
        expected_dist = (patch_attn * dist_matrix).sum(dim=-1)

        # Get average distance for each head
        mean_head_dist = expected_dist.mean(dim=-1)

        mean_distances.append(mean_head_dist)

    layers = np.arange(len(mean_distances))

    plt.figure(figsize=(10, 6))
    colors = distinctipy.get_colors(len(mean_distances[0]))
    for l in layers:
        heads_data = mean_distances[l].numpy()
        plt.scatter([l] * len(heads_data), heads_data, c=colors, s=15)

    plt.xlabel('Network depth (layer)')
    plt.ylabel('Mean attention distance (Pixels)')
    plt.ylim(bottom=0)
    plt.title(title)
    plt.legend()
    plt.grid()
    dir_name = os.path.dirname(__file__)
    file_name = os.path.join(dir_name, args.model + '_vit_locality')
    if vit_weights is not None:
        file_name += '_scratch'
    file_name += '.png'
    plt.savefig(file_name)
    plt.show()
