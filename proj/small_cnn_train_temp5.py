import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn.utils.prune as prune

from PIL import Image

import torch
import torchvision

from torchvision.transforms import Compose, PILToTensor, ToPILImage

# if torch.backends.mps.is_available():
#     device = 'mps'
# else:
#     device = 'cpu'

device = 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch.manual_seed(1)
torch.cuda.manual_seed(1)
net = Net().to(device)

training_required = True

if training_required:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(net, 'small_cnn_temp5/net.pt')

else:
    net = torch.load('small_cnn_temp5/net.pt')

# First Prune - 90%
encoder1_prune3 = type(net)().to(device)
encoder1_prune3.load_state_dict(net.state_dict())

print('Global Sparsity for Encoder: {:.2f}%'.format(
    100. * float(
        torch.sum(encoder1_prune3.conv1.weight == 0)
        + torch.sum(encoder1_prune3.conv2.weight == 0)
        + torch.sum(encoder1_prune3.fc1.weight == 0)
        + torch.sum(encoder1_prune3.fc2.weight == 0)
        + torch.sum(encoder1_prune3.fc3.weight == 0)
    )
    / float(encoder1_prune3.conv1.weight.nelement()
    + encoder1_prune3.conv2.weight.nelement()
    + encoder1_prune3.fc1.weight.nelement()
    + encoder1_prune3.fc2.weight.nelement()
    + encoder1_prune3.fc3.weight.nelement()
    )
))

parameters_to_prune_encoder = (
    (encoder1_prune3.conv1, 'weight'),
    (encoder1_prune3.conv2, 'weight'),
    (encoder1_prune3.fc1, 'weight'),
    (encoder1_prune3.fc2, 'weight'),
    (encoder1_prune3.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune_encoder,
    pruning_method = prune.L1Unstructured,
    amount = 0.9,
)

print('Global Sparsity for Encoder: {:.2f}%'.format(
    100. * float(
        torch.sum(encoder1_prune3.conv1.weight == 0)
        + torch.sum(encoder1_prune3.conv2.weight == 0)
        + torch.sum(encoder1_prune3.fc1.weight == 0)
        + torch.sum(encoder1_prune3.fc2.weight == 0)
        + torch.sum(encoder1_prune3.fc3.weight == 0)
    )
    / float(encoder1_prune3.conv1.weight.nelement()
    + encoder1_prune3.conv2.weight.nelement()
    + encoder1_prune3.fc1.weight.nelement()
    + encoder1_prune3.fc2.weight.nelement()
    + encoder1_prune3.fc3.weight.nelement()
    )
))

# First Prune - 90%
encoder1_prune4 = type(net)().to(device)
encoder1_prune4.load_state_dict(net.state_dict())

print('Global Sparsity for Encoder: {:.2f}%'.format(
    100. * float(
        torch.sum(encoder1_prune4.conv1.weight == 0)
        + torch.sum(encoder1_prune4.conv2.weight == 0)
        + torch.sum(encoder1_prune4.fc1.weight == 0)
        + torch.sum(encoder1_prune4.fc2.weight == 0)
        + torch.sum(encoder1_prune4.fc3.weight == 0)
    )
    / float(encoder1_prune4.conv1.weight.nelement()
    + encoder1_prune4.conv2.weight.nelement()
    + encoder1_prune4.fc1.weight.nelement()
    + encoder1_prune4.fc2.weight.nelement()
    + encoder1_prune4.fc3.weight.nelement()
    )
))

parameters_to_prune_encoder = (
    (encoder1_prune4.conv1, 'weight'),
    (encoder1_prune4.conv2, 'weight'),
    (encoder1_prune4.fc1, 'weight'),
    (encoder1_prune4.fc2, 'weight'),
    (encoder1_prune4.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune_encoder,
    pruning_method = prune.L1Unstructured,
    amount = 0.9,
)

print('Global Sparsity for Encoder: {:.2f}%'.format(
    100. * float(
        torch.sum(encoder1_prune4.conv1.weight == 0)
        + torch.sum(encoder1_prune4.conv2.weight == 0)
        + torch.sum(encoder1_prune4.fc1.weight == 0)
        + torch.sum(encoder1_prune4.fc2.weight == 0)
        + torch.sum(encoder1_prune4.fc3.weight == 0)
    )
    / float(encoder1_prune4.conv1.weight.nelement()
    + encoder1_prune4.conv2.weight.nelement()
    + encoder1_prune4.fc1.weight.nelement()
    + encoder1_prune4.fc2.weight.nelement()
    + encoder1_prune4.fc3.weight.nelement()
    )
))

training_required = True

# other_criterion = nn.CrossEntropyLoss()
if training_required:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(encoder1_prune3.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(25):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = encoder1_prune3(inputs)
            net_outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(other_loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(encoder1_prune3, 'small_cnn_temp5/prune1.pt')
else:
    encoder1_prune3 = torch.load('small_cnn_temp5/prune1.pt')

training_required = True

if training_required:

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(encoder1_prune4.parameters(), lr=0.0001, momentum=0.9)
    other_criterion = nn.KLDivLoss(reduction = 'mean')
    # other_criterion = nn.CrossEntropyLoss()
    for epoch in range(25):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = encoder1_prune4(inputs)
            net_outputs = net(inputs)
            loss = criterion(outputs, labels)
            other_loss = other_criterion(F.log_softmax(outputs/5), F.softmax(net_outputs/5))
            # print(other_loss)
            loss += other_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder1_prune4.parameters(), max_norm = 2, norm_type = 2)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(encoder1_prune4, 'small_cnn_temp5/prune2.pt')
else:
    encoder1_prune4 = torch.load('small_cnn_temp5/prune2.pt')

net.eval()
encoder1_prune3.eval()
encoder1_prune4.eval()


correct = 0
total = 0

print('TEST ON NO NOISE')

with torch.no_grad():
    for data in testloader:
        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('TEST ON NOISY DATA WITH STD OF 0.05')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.05**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.05**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.05**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('TEST ON NOISY DATA WITH STD OF 0.075')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.075**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.075**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.075**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


print('TEST ON NOISY DATA WITH STD OF 0.1')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.1**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.1**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.1**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


print('TEST ON NOISY DATA WITH STD OF 0.125')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.125**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.125**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.125**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


print('TEST ON NOISY DATA WITH STD OF 0.15')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.15**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.15**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.15**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


print('TEST ON NOISY DATA WITH STD OF 0.2')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.2**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.2**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune3(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        all_images = []
        for j in range(len(images)):
          torch.manual_seed(1)
          torch.cuda.manual_seed(1)
          img = images[j] + (0.2**0.5)*torch.randn(3, 32, 32)
          all_images.append(img)
        images = torch.stack(all_images)

        outputs = encoder1_prune4(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
