import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        dim_lat_space=4
        self.conv1=nn.Conv3d(1,32,kernel_size=3,stride=2,padding=1)
        self.bn1=nn.BatchNorm3d(32)
        self.conv2=nn.Conv3d(32,64,kernel_size=3,stride=2,padding=1)
        self.bn2=nn.BatchNorm3d(64)
        self.conv3=nn.Conv3d(64,128,kernel_size=3,stride=2,padding=1)
        self.bn3=nn.BatchNorm3d(128)
        self.conv4=nn.Conv3d(128,256,kernel_size=3,stride=2,padding=1)
        self.bn4=nn.BatchNorm3d(256)
        self.conv5=nn.Conv3d(256,512,kernel_size=3,stride=2,padding=1)
        self.bn5=nn.BatchNorm3d(512)
        self.fc=nn.Linear(512*2*2*2,dim_lat_space)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x)))
        x=F.relu(self.bn5(self.conv5(x)))
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        dim_lat_space=4
        self.fc=nn.Linear(dim_lat_space,512*2*2*2)
        self.deconv1=nn.ConvTranspose3d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn1=nn.BatchNorm3d(256)
        self.deconv2=nn.ConvTranspose3d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn2=nn.BatchNorm3d(128)
        self.deconv3=nn.ConvTranspose3d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn3=nn.BatchNorm3d(64)
        self.deconv4=nn.ConvTranspose3d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn4=nn.BatchNorm3d(32)
        self.deconv5=nn.ConvTranspose3d(32,1,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self,x):
        x=self.fc(x)
        x=x.view(x.size(0),512,2,2,2)
        x=F.relu(self.bn1(self.deconv1(x)))
        x=F.relu(self.bn2(self.deconv2(x)))
        x=F.relu(self.bn3(self.deconv3(x)))
        x=F.relu(self.bn4(self.deconv4(x)))
        x=torch.sigmoid(self.deconv5(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
