# train.py

import argparse
import torch
import torch
import torchvision.models as models
from torch import nn, optim


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')

parser.add_argument('data_directory', type=str, help='Directory of the data set')
parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints', default='.')
parser.add_argument('--arch', type=str, help='Model architecture', default='vgg16')
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=512)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=20)
parser.add_argument('--gpu', action='store_true', help='Use GPU for training', default=True)

args = parser.parse_args()

data_dir = args.data_directory
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
if gpu == True
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print('Device: ',device)
def build_model(arch='vgg16', hidden_units=512, learning_rate=0.001, gpu=False):
    if arch not in ['vgg16', '<other_archs>']:
        raise ValueError('Unsupported architecture.')

    if arch == 'VGG':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),
            nn.LogSoftmax(dim=1)
        )

    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(1024, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096, 102),
                        nn.LogSoftmax(dim=1)
                    )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Use GPU if it's available and "gpu" flag is True
    
    model.to(device)

    return model, optimizer, criterion


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    return dataloaders, image_datasets


def train_model(model, optimizer, criterion, epochs, dataloaders, use_gpu):
    if use_gpu:
        model = model.to('cuda')

    # Train the classifier layers using backpropagation
    epochs = 3
    steps = 0
    running_loss = 0
    print_every = 5
    model.to(device)

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
def main():
    
    # Load and transform the dataset
    dataloaders, image_datasets = load_data(data_dir)
    
    # Building and training the network
    model, optimizer, criterion = build_model(arch, hidden_units, lr, gpu)
    train_model(model, optimizer, criterion, epochs, dataloaders, gpu)
    
    # Save the checkpoint
    save_checkpoint(model, optimizer, image_datasets['train'], save_dir, arch)
 

if __name__ == '__main__':
    main()