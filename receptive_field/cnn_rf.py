import matplotlib.pyplot as plt
import numpy as np
import os


def get_resnet_max_distances_basic_block(layers: list[int]):
    # Initial layer -> 7x7 convolution with 2x2 stride
    horizontal_max_distances = [6]
    current_max_distance = 6
    current_stride = 2

    # Max pool -> 3x3 kernel with 2x2 stride
    current_max_distance += (2 * current_stride)
    horizontal_max_distances.append(current_max_distance)
    current_stride *= 2

    for i in range(len(layers)):
        layer = layers[i]
        for j in range(layer):
            # Two Convolutions with 3x3 kernels.
            if i > 0 and j == 0:
                # For layers 2-4, there is a 2x2 stride in the first block's first convolution.
                current_max_distance += (2 * current_stride)
                current_stride *= 2
                current_max_distance += (2 * current_stride)
            else:
                current_max_distance += (4 * current_stride)
            horizontal_max_distances.append(current_max_distance)

    return [np.sqrt(2) * min(h, 283) for h in horizontal_max_distances]

def get_resnet_max_distances_bottleneck(layers: list[int]):
    # Initial layer -> 7x7 convolution with 2x2 stride
    horizontal_max_distances = [6]
    current_max_distance = 6
    current_stride = 2

    # Max pool -> 3x3 kernel with 2x2 stride
    current_max_distance += (2 * current_stride)
    horizontal_max_distances.append(current_max_distance)
    current_stride *= 2

    for i in range(len(layers)):
        layer = layers[i]
        for j in range(layer):
            # Single convolution with 3x3 kernel
            current_max_distance += (2 * current_stride)
            horizontal_max_distances.append(current_max_distance)
            if i > 0 and j == 0:
                # For layers 2-4, there is a 2x2 stride in the first block's first convolution.
                current_stride *= 2

    return [np.sqrt(2) * min(h, 283) for h in horizontal_max_distances]

def plot(distances, title):
    dir_name = os.path.dirname(__file__)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.scatter(np.arange(len(distances)), distances, s=15)
    plt.xlabel('Network depth (layer)')
    plt.ylabel('Maximum Intra-Kernel Distance (Pixels)')
    plt.grid()
    plt.savefig(os.path.join(dir_name, title + '.png'))
    plt.show()

if __name__ == '__main__':
    # These are pulled directly from torchvision's resnet.py
    # These lists are passed directly into the constructor for the Resnet class.
    resnet18_layers = [2, 2, 2, 2]
    resnet34_layers = [3, 4, 6, 3]
    resnet50_layers = [3, 4, 6, 3]
    resnet101_layers = [3, 4, 23, 3]
    resnet152_layers = [3, 8, 36, 3]

    # 18 and 34 are built using basic blocks.
    resnet18_distances = get_resnet_max_distances_basic_block(resnet18_layers)
    resnet34_distances = get_resnet_max_distances_basic_block(resnet34_layers)
    # The rest are built using bottleneck blocks, which have fewer weights
    resnet50_distances = get_resnet_max_distances_bottleneck(resnet50_layers)
    resnet101_distances = get_resnet_max_distances_bottleneck(resnet101_layers)
    resnet152_distances = get_resnet_max_distances_bottleneck(resnet152_layers)

    plot(resnet18_distances, 'ResNet18')
    plot(resnet34_distances, 'ResNet34')
    plot(resnet50_distances, 'ResNet50')
    plot(resnet101_distances, 'ResNet101')
    plot(resnet152_distances, 'ResNet152')

