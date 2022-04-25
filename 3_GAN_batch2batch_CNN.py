from cmath import tanh
from tkinter.tix import Tree
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
EPOCH = 100
BATCHSIZE = 4
LEARNINGRATE = 0.0003
D_STEPS_PER_G = 1 # times for discriminator trainning during a trainning in generator

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),

])

data_train = datasets.MNIST(root='./data/', transform=transform, train=True, download=True)
data_test = datasets.MNIST(root='./data/', transform=transform, train=False)

dataloader_train = DataLoader(dataset=data_train, batch_size=BATCHSIZE, shuffle=True)
dataloader_test = DataLoader(dataset=data_test, batch_size=BATCHSIZE, shuffle=True)

def save_batch(batch_tensor, path):
    img = torchvision.utils.make_grid(batch_tensor, padding = 1)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # change【3，32，128】-->【32,128,3】
    plt.savefig(path)

''' 
Generator class:
    input:  Noise Tensor in (Batchsize, 1, 100)
    output: (Batchsize,1, 28, 28)
'''
class generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(100, 3136)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, 1, 1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, 2),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

''' 
Discriminator class:
    input:  Image tensor in (Batchsize, 1, 28,28)
    output: (1)
        output is only a number in 1 or 0, represent to whether the image is true Trainning data or generated image.
'''
class discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        x = x.view(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

    
# Instantiate the generator and discriminator
G = generator()
D = discriminator()
if torch.cuda.is_available():
    G = G.cuda()
    D = D.cuda()

# Use BCE loss function to represent the formula in the GAN paper
criterion = nn.BCELoss()
# Define the optimizers of generator and optimizer respectively
G_optimizer = torch.optim.Adam(G.parameters(), LEARNINGRATE)
D_optimizer = torch.optim.Adam(D.parameters(), LEARNINGRATE)

def train_together():
    for epoch in range(EPOCH):
        print("====EPOCH {}=====".format(epoch))
        for i, (img, _) in enumerate(dataloader_train):
            num_img = img.size(0)
            # =================train discriminator
            img = img.view(num_img, -1)
            real_img = Variable(img).cuda()
            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()
    
            # compute loss of real_img
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better
    
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, 100)).cuda()
            fake_img = G(z)
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better
    
            # bp and optimize
            d_loss = d_loss_real + d_loss_fake
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()
            
             # ===============train generator
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, 100)).cuda()
            fake_img = G(z)
            output = D(fake_img)
            
            # !!!Attention! Here is the loss between REAL_LABEL and fake image
            # When optiminze, this loss means to make the fake images more similar to the real images. 
            # Equal to imporve the ability of Generator.
            g_loss = criterion(output, real_label) 
            
            # bp and optimize
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
    
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                    'D real: {:.6f}, D fake: {:.6f}'.format(
                        epoch, EPOCH, d_loss, g_loss,
                        real_scores.data.mean(), fake_scores.data.mean()))

        noise = torch.randn(64,1,100).cuda()
        save_batch(G(noise).cpu(), "res{}.jpg".format(epoch))
        print("--img saved--")
        
if __name__ == '__main__':
    train_together()