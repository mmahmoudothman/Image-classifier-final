import sys
import argparse
from collections import OrderedDict
from utils import *

def load_train_data(train_dir, batch_size):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(300),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size = batch_size, shuffle = True)
    return train_loader, train_datasets.class_to_idx

def load_valid_data(valid_dir, batch_size):

    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    valid_loader = torch.utils.data.DataLoader(
        valid_datasets, batch_size = batch_size, shuffle = False)
    
    return valid_loader

def save_checkpoint(file_name, net, arch, class_to_idx, num_epochs):
    torch.save(
        {'num_epochs': num_epochs,
        'arch': arch,
        'state_dict': net.state_dict(),
        'train_datasets_class_to_idx':class_to_idx
        },
        file_name)
    print("save_checkpoint completed")

def build_network(arch, hidden_size):
    if arch == 'vgg16':
        net = models.vgg16(pretrained=False)
    elif arch == 'vgg13':
        net = models.vgg13(pretrained=False)
    elif arch == 'vgg11':
        net = models.vgg11(pretrained=False)
    elif arch == 'vgg19':
        net = models.vgg19(pretrained=False)
    else:
        net = models.vgg16(pretrained=False)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(224*112, hidden_size)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_size, 512)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))  
    net.classifier = classifier
    for param in net.parameters():
        param.requires_grad = False 
    for param in net.classifier.parameters():
        param.requires_grad = True 
        
    return net

def loss_funcation(net, lr_input):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = lr_input, momentum = 0.9)
    
    return criterion, optimizer

def train(net, criterion, optimizer, num_epochs, train_loader, batch_size, cuda_enabled):
    for epoch in range(num_epochs):
        batch_size_start = time.time()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if cuda_enabled:
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs = Variable(inputs)
            target = Variable(labels)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            per_num = 50
            if i % per_num == (per_num - 1):
                print('Epoch [%d/%d/%d], Loss: %.4f,need time %.4f'
                  % (i/per_num, epoch + 1, num_epochs, running_loss / (per_num / batch_size), time.time() - batch_size_start))
                running_loss = 0.0
        print('Epoch [%d/%d],Time [%.4f], Finished Training'
              % (epoch + 1, num_epochs, time.time() - batch_size_start))
    return net, running_loss

def validate(net, cuda_enabled, valid_loader):
    correct = 0
    total = 0
    net.eval()
    for i, (images, labels) in enumerate(valid_loader):   
        val_input = Variable(images, True)
        val_label = Variable(labels.long(), True)
        if cuda_enabled:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        outputs = net(val_input)
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += val_label.size(0)
        correct += (predicted == val_label.cpu().long()).sum()
        
    accuracy = float(100 * correct) / float(total)
    print("Accuracy: [%d / %d] = %.4f \n" 
          % (correct, total, accuracy))
    net.train()
    
    return accuracy

def main(argv): 
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', metavar='train_dir', nargs='+', help='path to load training data')
    parser.add_argument('--valid_dir', help='path to load valid data')
    parser.add_argument('--save_dir', help='path for saving trained model')
    parser.add_argument('--arch', help='network type, supports vgg11, vgg13, vgg16, vgg19. By default vgg16')
    parser.add_argument('--learning_rate', help='learning rate, such as 0.01, by default 0.001', type = float)
    parser.add_argument('--hidden_units', help='hidden units, such as 512, by default 128', type = int)
    parser.add_argument('--epochs', help='training epochs, 1,2..., by default 1 ', type = int)
    parser.add_argument('--gpu', help='enable gpu or not, 1 for enabled, otherwise disabled, by default 0', type = int)
    args = parser.parse_args()    
    
    batch_size = 4
    if args.train_dir:
        train_loader, class_to_idx = load_train_data(args.train_dir[0], batch_size)
    else:
        print(' train_dir is missing')
        return 
    
    cuda_enabled = False
    if args.gpu:
        if(args.gpu == 1 and torch.cuda.is_available()):
            cuda_enabled = True
    print("cuda_enabled=\n", cuda_enabled)
    arch = 'vgg16'
    if args.arch:
        arch = args.arch
    hidden_units = 128
    if args.hidden_units:
        hidden_units = args.hidden_units 
    net = build_network(arch, hidden_units)
    if cuda_enabled:
        net.cuda()
    print(net)
    
    lr_input = 0.001
    if args.learning_rate:
        lr_input = args.learning_rate
    criterion, optimizer = loss_funcation(net, lr_input)
    num_epochs = 1
    if args.epochs:
        num_epochs = args.epochs
    net, loss = train(net, criterion, optimizer, num_epochs, train_loader, batch_size, cuda_enabled)

    if args.save_dir:
        save_checkpoint(args.save_dir, net,arch, class_to_idx, num_epochs)
    if args.valid_dir:
        valid_loader = load_valid_data(args.valid_dir, batch_size)
        accuracy = validate(net, cuda_enabled, valid_loader)

if __name__ == '__main__':
    main(sys.argv)