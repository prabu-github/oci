from __future__ import print_function
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='data_dir')
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        default=64, 
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', 
                        type=int, 
                        default=100, 
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=2, 
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', 
                        type=float, 
                        default=1.0, 
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.7, 
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--device',
                        type=str,
                        action='store', 
                        default='cpu',
                        help='which device')
    parser.add_argument('--seed', 
                        type=int, 
                        default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--model_dir',
                        type=str,
                        action='store', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    
    device = args.device
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(device, model_dir)
    

    train_kwargs = {'batch_size': args.train_batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if 'cuda' in device:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
   
    print(train_kwargs)
    print(test_kwargs)
    
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST(data_dir, 
                              train=True, 
                              download=False,
                              transform=transform)
    dataset2 = datasets.MNIST(data_dir, 
                              train=False,
                              download=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print(len(train_loader.dataset), len(test_loader.dataset))

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    model_file = f'mnist_model_lr{args.lr:1.3f}_nepochs{args.num_epochs}_seed{args.seed}.pt'
    torch.save(model.state_dict(), model_dir/model_file)


if __name__ == '__main__':
    main()
