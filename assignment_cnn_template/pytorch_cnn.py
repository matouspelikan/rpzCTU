import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(in_features=28*28,
                            out_features=128)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28*28*10//(2*2),
                            out_features=128)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError("You have to implement this function.")
        return output

def classify(model, x):
    '''
    :param model:    network model object
    :param x:        (batch_sz, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_sz, ) torch tensor with class labels
    '''
    raise NotImplementedError("You have to implement this function.")
    return labels

def get_model_class(_):
    ''' Do not change, needed for AE '''
    return [MyNet]


def train():
    batch_sz = 64
    learning_rate = 1
    epochs = 15
    
    dataset = datasets.MNIST('data', train=True, download=True,
                             transform=transforms.ToTensor())


    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                      [train_size,
                                                                       val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_sz,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=batch_sz,
                                             shuffle=True)

    device = torch.device("cpu")
    model = FCNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        for i_batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            net_output = model(x)
            loss = F.nll_loss(net_output, y)
            loss.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                print('Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
                    epoch, i_batch, loss.item()))

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                net_output = model(x)

                prediction = classify(model, x)
                correct += prediction.eq(y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        print('Validation accuracy: {:.2f}%'.format(100 * val_accuracy))

        torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    train()
