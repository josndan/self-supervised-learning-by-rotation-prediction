import torch
import torch.nn as nn
from utils import *
from torchvision import models


def load_cuda(model,load_from):
  if torch.cuda.is_available():
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')
      
  if load_from is not None:
    model.load_state_dict(torch.load(load_from,map_location=device))

  if torch.cuda.is_available():
    model.cuda()
  
  return model

def get_six_layer_net(input_size, num_classes,load_from = None):
  channel_1 = 32
  channel_2 = 16
  channel_4 = 16
  channel_5 = 10
  hidden_layer_size = 200

  H1 = 1 + (input_size + 2 * 2 - 5)
  W1 = 1 + (input_size + 2 * 2 - 5)
  H2 = 1 + (H1 + 2 * 2 - 5)
  W2 = 1 + (W1 + 2 * 2 - 5)
  H3 = 1 + (H2 + 2 * 2*0 - 3)//3
  W3 = 1 + (W2 + 2 * 2*0 - 3)//3

  H4 = 1 + (H3 + 2 * 1 - 3)
  W4 = 1 + (W3 + 2 * 1 - 3)
  H5 = 1 + (H4 + 2 * 1 - 3)
  W5 = 1 + (W4 + 2 * 1 - 3)
  H6 = 1 + (H5 + 2 * 2*0 - 3)//3
  W6 = 1 + (W5 + 2 * 2*0 - 3)//3


  net = nn.Sequential(nn.BatchNorm2d(3),nn.ReLU(),nn.Conv2d(3,channel_1,5,padding = 2),
                        nn.BatchNorm2d(channel_1),nn.ReLU(),nn.Conv2d(channel_1,channel_2,5,padding = 2),
                        nn.BatchNorm2d(channel_2), nn.ReLU(),nn.MaxPool2d(3),
                        nn.BatchNorm2d(channel_2),nn.ReLU(),nn.Conv2d(channel_2,channel_4,3,padding = 1),
                        nn.BatchNorm2d(channel_4),nn.ReLU(),nn.Conv2d(channel_4,channel_5,3,padding = 1),
                        Flatten(),nn.Linear(channel_5*H5*W5, hidden_layer_size),
                        nn.Linear(hidden_layer_size, num_classes))

  net.save_name = "six_layer_net"
  return load_cuda(net,load_from)


def get_efficient_net(input_size, num_classes,load_from = None):
  net = models.efficientnet_b0(pretrained=False,num_classes = num_classes)
  net.save_name = "efficient_net"

  return load_cuda(net,load_from)


def get_mobile_net(input_size, num_classes,load_from = None):
  net = models.mobilenet_v2(pretrained=False,num_classes = num_classes)
  net.save_name = "mobile_net"
  return load_cuda(net,load_from)


