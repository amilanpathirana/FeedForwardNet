
print("Hi")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

##device=torch.device("cuda" if torch.cuda.is_available else "cpu")
device=torch.device("cpu")

input_size=784
hidden_size=500
output_size=10
num_epochs=2
batch_size=100
learnning_rate=0.001

train_dataset=torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset=torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)

test_loader=torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True)


examples=iter(train_loader)
samples,labels=examples.next()
print(samples.shape,labels.shape)


for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap="gray")
#plt.show()


class MyRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.myrelu1=nn.ReLU()
       
    def forward(self, x):
        out1=self.myrelu1(x)
        return out1





class FFNet(nn.Module):
    def __init__(self,input_size,output_size, hidden_size):
        super(FFNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.lh=nn.Linear(hidden_size,hidden_size)
        self.l2=nn.Linear(hidden_size,output_size)
        self.lr=nn.Linear(output_size,hidden_size)
        self.relu2=MyRelu()
        
        
        

    def forward(self,x):
        out1=self.l1(x)
        out2=self.relu2(out1)
        out3=self.lh(out2)
        out4=self.relu2(out3)
        out5=self.l2(out4)

        outr=out1-self.lr(out5)


        #out6=self.l1(out5)
        out7=self.relu2(outr)
        out8=self.lh(out7)
        out9=self.relu2(out8)
        out10=self.l2(out9)

        #outr2=out7-self.lr(out11)
        return out10


model=FFNet(input_size,output_size,hidden_size)



#Loss and Optimizer

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learnning_rate)


#Trainning Loop

steps=len(train_loader)

for epoch in range (num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #image size is 100x1x28x28   , need to convert this to 100x784
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(i+1)%100==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {steps}, loss= {loss.item():.4f}')



#Test

with torch.no_grad():
    n_correct=0
    n_samples=0

    for (images,labels) in test_loader:
        images=images.reshape(-1,28*28).to(device)
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        n_samples+=labels.shape[0]
        n_correct+=(predictions==labels).sum().item()

    acc=100.0*n_correct/n_samples
    print(f"accuracy = {acc}")




























    















