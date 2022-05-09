import minist
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

continued_network = minist.Network().to(device)
continued_optimizer = optim.SGD(continued_network.parameters(), lr=minist.learning_rate,
                                momentum=minist.momentum)

network_state_dict = torch.load('results/model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

for i in range(4,9):
  test_counter.append(i*len(train_loader.dataset))
  train(i)
  test()