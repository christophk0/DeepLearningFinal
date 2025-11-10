import torch
from torchvision import datasets, transforms
import yaml

from VisionTransormer import VisionTransformer
from CNN import CNN

if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = yaml.safe_load(open('config.yaml', 'r'))
    num_epochs = config['num_epochs']
    if config['model_type'] == 'vt':
        model = VisionTransformer(config=config['vision_transformer'], device=device)
    else:
        model = CNN(config=config['cnn'], device=device)

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    training_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=config['batch_size'],
                                                  shuffle=True)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(training_loader):
            model.training_step(batch, epoch, batch_idx, print_debug=(batch_idx % 50 == 0))
        test_loss = 0
        correct = 0
        for batch_idx, batch in enumerate(test_loader):
            test_result = model.test_step(batch)
            test_loss += test_result[0]
            correct += test_result[1]
        test_loss /= len(test_loader)
        correct /= len(test_data)
        print(f"Test Error for Epoch {epoch}: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        test_loss = 0
        correct = 0

