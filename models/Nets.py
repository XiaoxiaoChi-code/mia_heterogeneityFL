from torch import nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class Mnistcnn(nn.Module):
    def __init__(self, args):
        super(Mnistcnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CIFAcnn(nn.Module):
    def __init__(self, args):
        super(CIFAcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3*32*32 ==》6*28*28(32 + 2P - kernel + 1)
        self.pool = nn.MaxPool2d(2, 2)  # 6*28*28 ==> 6*14*14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6*14*14 ==> 16 * 10 * 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 还要一个pooling 所以输入是16 * 5 * 5 ==> 120
        self.fc2 = nn.Linear(120, 84)  # 120 ==> 84
        self.fc3 = nn.Linear(84, args.num_classes)  # 84 ==> 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
