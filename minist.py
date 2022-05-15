import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import datasets, transforms


# 定义网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=40 * 4 * 4, out_features=1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 2*2的核，步长为2，pooling之后的大小除以2

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 40 * 4 * 4)  # 展开为行向量

        x = F.relu(self.fc1(x))
        x = F.dropout(input=x, p=0.5, training=self.training)  # 弃权， 一半的神经元

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)  # 按行进行log(softmax(x))
        return x


# 数据加载
def data_loader(batch_size, batch_size_test, use_cuda=False):
    """
    数据加载器

    :param batch_size: 训练集批次大小
    :param batch_size_test:  测试集批次大小
    :param use_cuda: 是否使用GPU
    :return: 训练集和测试集
    """

    # GPU训练需要的参数
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # 数据处理器
    transform = transforms.Compose([
        # 把[0,255]的(H,W,C)的图片转换为[0,1]的(channel,height,width)的图片
        transforms.ToTensor(),
        # z-score标准化为标准正态分布
        # 这两个数分别是MNIST的均值和标准差
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data/',
                       train=True,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data/',
                       train=False,
                       transform=transform),
        batch_size=batch_size_test,
        shuffle=True,
        **kwargs)

    return train_loader, test_loader


# 训练脚本
def train(log_interval, network, device, train_loader, batch_size_train, train_losses, train_counter, optimizer, epoch):
    """
    训练过程

    :param log_interval: 计算训练效果的间隔
    :param network: 网络模型
    :param device: 使用的设备 CPU or GPU
    :param train_loader: 训练集
    :param batch_size_train: 批次大小
    :param train_losses: 记录损失变化
    :param train_counter: 记录训练次数
    :param optimizer: 优化器
    :param epoch: 当前的迭代次数
    :return: null
    """
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 每个batch重新计算梯度
        optimizer.zero_grad()
        # 前向计算出预测输出
        output = network(data)
        # 对数似然代价
        loss = F.nll_loss(output, target)
        # 求梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 每经过一个log_interval大小的间隔，记录一下训练效果
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            # torch.save(network.state_dict(), 'results/model.pth')
            # torch.save(optimizer.state_dict(), 'results/optimizer.pth')


# 测试
def test(network, device, test_loader, test_losses, accuracies=None):
    """
    测试过程

    :param accuracies: 记录每次迭代的精确度
    :param network: 网络模型
    :param device: 测试使用的设备
    :param test_loader: 测试集
    :param test_losses: 记录损失变化
    :return: null
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 预测时不需要反向传播
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()

    # 上面test_loss得到的是累加和，这里求得均值
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracies.append(100. * correct / len(test_loader.dataset))


def drawFig(train_counter, train_losses, test_counter, test_losses):
    """
    绘图

    :param train_counter:
    :param train_losses:
    :param test_counter:
    :param test_losses:
    :return:
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_counter[:len(train_losses)], train_losses, color='blue')
    plt.scatter(test_counter[:len(test_losses)], test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == '__main__':
    n_epochs = 20
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1

    # 启用英伟达cuDNN加速框架和CUDA
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}...".format("cuda" if use_cuda else "cpu"))

    # 加载数据
    train_loader, test_loader = data_loader(batch_size=batch_size_train, batch_size_test=batch_size_test,
                                            use_cuda=use_cuda)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    accuracies = []
    max_acc = 0.
    max_interval = 3

    network = Network().to(device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    test(network, device, test_loader, test_losses, accuracies)
    max_acc = max(max_acc, accuracies[-1])
    for epoch in range(1, n_epochs + 1):
        train(log_interval, network, device, train_loader, batch_size_train, train_losses, train_counter, optimizer,
              epoch)
        test(network, device, test_loader, test_losses, accuracies)

        max_acc = max(max_acc, accuracies[-1])
        print('Max accuracy: {:.2f}%\n'.format(max_acc))
        if max(accuracies[-max_interval:]) < max_acc:
            print('No progress, stop training.')
            break

    drawFig(train_counter, train_losses, test_counter, test_losses)

    # if input('Continue training?(y/n)') == 'y':
    #     continued_network = Network()
    #     continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
    #                                     momentum=momentum)
    #     network_state_dict = torch.load('results/model.pth')
    #     continued_network.load_state_dict(network_state_dict)
    #
    #     optimizer_state_dict = torch.load('results/optimizer.pth')
    #     continued_optimizer.load_state_dict(optimizer_state_dict)
    #
    #     for epoch in range(4, 9):
    #         test_counter.append(epoch * len(train_loader.dataset))
    #         train(log_interval, network, device, train_loader, batch_size_train, train_losses, train_counter, optimizer,
    #               epoch)
    #         test(network, device, test_loader, test_losses)
    #
    #     drawFig(train_counter, train_losses, test_counter, test_losses)
