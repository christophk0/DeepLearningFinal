import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import vit_b_16, vit_l_16, vit_h_14, vit_b_32, resnet18, resnet50, resnet152

from cka import CKACalculator, export_cka_plot

plt.rcParams['figure.figsize'] = (7, 7)

def get_model(model_name, weights_path=None, device='cpu'):
    if weights_path is not None:
        model_weights = torch.load(weights_path)
    name = model_name
    if model_name == 'vit_b_16':
        if weights_path is None:
            model_weights = models.ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=model_weights)
        name = 'VIT-B/16'
    elif model_name == 'vit_l_16':
        if weights_path is None:
            model_weights = models.ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=model_weights)
        name = 'VIT-L/16'
    elif model_name == 'vit_h_14':
        if weights_path is None:
            model_weights = models.ViT_H_14_Weights.DEFAULT
        model = vit_h_14(weights=model_weights)
        name = 'VIT-H/14'
    elif model_name == 'vit_b_32':
        if weights_path is None:
            model_weights = models.ViT_B_32_Weights.DEFAULT
        model = vit_b_32(weights=model_weights)
        name = 'VIT-B/32'
    elif model_name == 'resnet18':
        if weights_path is None:
            model_weights = models.ResNet18_Weights.DEFAULT
        model = resnet18(weights=model_weights)
        name = 'ResNet-18'
    elif model_name == 'resnet50':
        if weights_path is None:
            model_weights = models.ResNet50_Weights.DEFAULT
        model = resnet50(weights=model_weights)
        name = 'ResNet-50'
    elif model_name == 'resnet152':
        if weights_path is None:
            model_weights = models.ResNet152_Weights.DEFAULT
        model = resnet152(weights=model_weights)
        name = 'ResNet-152'
    else:
        raise NotImplementedError('Unsupported model type')
    
    model.to(device)
    model.eval()
    
    return model, name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='resnet50')
    parser.add_argument('--model2', type=str, default='resnet18')
    parser.add_argument('--weight1', type=str, default=None)
    parser.add_argument('--weight2', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--exportpath', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Dataloader. TODO: Expand to other datasets
    batch_size = args.batchsize
    transforms = Compose([Resize(224),ToTensor(), 
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = CIFAR10(root='../data/', train=False, download=True, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # Initialize models
    model_1,name_1 = get_model(args.model1,args.weight1, device=device)
    model_2,name_2 = get_model(args.model2,args.weight2, device=device)

    # CKA setup and computation
    calculator = CKACalculator(model1=model_1, model2=model_2, dataloader=dataloader, num_epochs = args.epochs)

    cka_output = calculator.calculate_cka_matrix()
    print(f"CKA output size: {cka_output.size()}")

    # Show image and export
    cka_matrix = cka_output.cpu().numpy()
    export_path = f"CKA_{name_1.replace('/', '')}_{name_2.replace('/', '')}_epochs_{args.epochs}.png" if args.exportpath is None else args.exportpath
    
    export_cka_plot(cka_matrix, name_1, name_2, save_path=export_path)