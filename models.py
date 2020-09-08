
from my_tools import *
from utils import A, At

import torch
import torch.nn as nn


class ADMM_net(nn.Module):

    def __init__(self):
        super(ADMM_net, self).__init__()
                
        self.unet1 = Unet(8, 8)
        self.unet2 = Unet(8, 8)
        self.unet3 = Unet(8, 8)
        self.unet4 = Unet(8, 8)
        self.unet5 = Unet(8, 8)
        self.unet6 = Unet(8, 8)
        self.unet7 = Unet(8, 8)
        self.unet8 = Unet(8, 8)
        self.unet9 = Unet(8, 8)   
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s):
        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        theta = self.unet1(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        theta = self.unet2(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        theta = self.unet3(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        theta = self.unet4(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        theta = self.unet5(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        x1 = x-b
        theta = self.unet6(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        x1 = x-b
        theta = self.unet7(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma8),Phi)
        x1 = x-b
        theta = self.unet8(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma9),Phi)
        x1 = x-b
        theta = self.unet9(x1)
        b = b- (x-theta)
        x_list.append(theta)

        output_list = x_list[-3:]
        return output_list