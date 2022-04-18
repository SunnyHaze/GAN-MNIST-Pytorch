import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
EPOCH = 100
BATCHSIZE = 128
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
        self.layer = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    def forward(self, x:tensor):
        return self.layer(x).view((x.shape[0],1, 28, 28))

''' 
Discriminator class:
    input:  Image tensor in (Batchsize, 1, 28,28)
    output: (1)
        output is only a number in 1 or 0, represent to whether the image is true Trainning data or generated image.
'''
class discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        ) 
    def forward(self, x):
        x1 = x.view((x.shape[0], 784))
        return self.layer(x1).squeeze(1)
    
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

# Define a process of a trainning in discriminator in 1 Epoch
def train_discriminator():
    for id,(true_img, label) in enumerate(dataloader_train):
        true_labels = torch.ones(len(label))
        fake_labels = torch.zeros(len(label))
        noise = torch.randn(len(label), 1, 100)
        # =============CUDA==========
        if torch.cuda.is_available():
            true_labels = true_labels.cuda()
            fake_labels = fake_labels.cuda()
            true_img = true_img.cuda()
            noise = noise.cuda()
        # ===========================
        fake_img = G(noise)
        # True img loss
        predict = D(true_img)
        true_loss = criterion(predict, true_labels)
        # Fake img loss
        predict = D(fake_img)
        fake_loss = criterion(predict, fake_labels)
        # backpropagation and Optimize 
        total_loss = true_loss + fake_loss
        D_optimizer.zero_grad()
        total_loss.backward()
        D_optimizer.step()
        
        if id % int(len(dataloader_train) / 3) == 0:
            print("  Discriminator-trainning   Epoch:[{}/{}]  loss: {:.7f}". format(id, EPOCH, fake_loss))
    return total_loss

# Define a process of a trainning in generator in 1 Epoch
def train_generator():
    for i in range(len(dataloader_train)):
        # randn initialized noise to genrate the images
        noise = torch.randn(BATCHSIZE, 1, 100) 
        real_label = torch.ones(BATCHSIZE)
        if torch.cuda.is_available():
            noise = noise.cuda()
            real_label = real_label.cuda()
        fake_img = G(noise)
        predict = D(fake_img)
        fake_loss = criterion(predict, real_label) # !!!Attention! here is the loss between REAL_LABEL and fake image
        # When optiminze, this loss means to make the fake images more similar with the real images. 
        G_optimizer.zero_grad()
        fake_loss.backward()
        G_optimizer.step()
        if i % int(len(dataloader_train) / 3) == 0:
            print("  Generator-trainning   Epoch:[{:>3d}/{}]  loss: {:.7f}". format(i, EPOCH, fake_loss))
    return fake_loss

if __name__ == '__main__':
    for i in range(EPOCH):
        print("========Epoch{} start=======".format(i))
        D_loss = 0
        G_loss = 0
        for k in range(D_STEPS_PER_G):
            print("--\"D\" Trarining [{}/{}]--".format(k, D_STEPS_PER_G))
            D_loss = train_discriminator()
            print("-" * 20)
        print("--\"G\" Trarining [{:>3d}/{}]--".format(k, D_STEPS_PER_G))
        G_loss = train_generator()
        print("-"*20)
        noise = torch.randn(64,1,100).cuda()
        save_batch(G(noise).cpu(), "res{}.jpg".format(i))
        print("========Epoch{} Done=======".format(i))
