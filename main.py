import torch
from torchvision import datasets, transforms
import yaml
import pprint

from VisionTransormer import VisionTransformer
from CNN import CNN

if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = yaml.safe_load(open('config.yaml', 'r'))
    pprint.pprint(config)
    num_epochs = config['num_epochs']
    if config['model_type'] == 'vt':
        model = VisionTransformer(config=config['vision_transformer'], device=device)
    else:
        model = CNN(config=config['cnn'], device=device)

    if config['skip_train_and_test']:
        exit("Skipping train and test due to skip_train_and_test config")

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    training_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=config['batch_size'],
                                                  shuffle=True)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)

    save_checkpoint = config['save_checkpoint_path']

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(training_loader):
            loss = model.training_step(batch)
            if batch_idx % config['print_batch_frequency'] == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        test_loss = 0
        correct = 0
        for batch_idx, batch in enumerate(test_loader):
            test_result = model.test_step(batch)
            test_loss += test_result[0]
            correct += test_result[1]
        test_loss /= len(test_loader)
        correct /= len(test_data)
        print(f"Test Error for Epoch {epoch}: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        if save_checkpoint != '':
            torch.save(model.state_dict(), f'epoch_{epoch}_'+save_checkpoint)
            print("Checkpoint saved to epoch_{epoch}_{save_checkpoint}")

    save_weights = config['save_weights_path']
    if save_weights != '':
        torch.save(model.state_dict(), save_weights)
        print("Weights saved to {save_weights}")
