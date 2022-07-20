from matplotlib.pyplot import axis
import torch 
import torchvision.transforms as T
from utils import sample_spherical
from torch.utils.data import Dataset
import numpy as np

def identity_transfrom(inputs,y):
  return inputs,y

def create_rotate_supervision(inputs,dummy,bbox=None):
  res = []
  labels = []
  if bbox:
    res_bbox = []
  for i,angle in enumerate([0,90,180,270]):
    res.append(T.functional.rotate(inputs,angle))
    if bbox:
      res_bbox.append(bbox)
    labels.append(torch.full((inputs.shape[0],),i))
  
  res = torch.cat(res,dim=0)
  labels = torch.cat(labels,dim=0)
  if bbox:
    res_bbox = np.concatenate(res_bbox,axis=0)
  N = labels.shape[0]
  prem = torch.randperm(N)
  
  if bbox:
    return res[prem],labels[prem],res_bbox[prem]
  else:
    return res[prem],labels[prem]

def create_color_supervision(inputs,dummy):
  res = []
  labels = []
  for i,angle in enumerate([0,90,180,270]):
    res.append(T.functional.rotate(inputs,angle))
    labels.append(torch.full((inputs.shape[0],),i))
  
  res = torch.cat(res,dim=0)
  labels = torch.cat(labels,dim=0)
  N = labels.shape[0]
  prem = torch.randperm(N)
  
  return res[prem],labels[prem]

def create_context_supervision(inputs,dummy):
  N,C,H,W = inputs.shape
  sH = H - (H%3)
  sW = W - (W%3)
  inputs = T.functional.resize(inputs,(sH,sW))
  N,C,H,W = inputs.shape
  sH,sW = sH//3,sW//3
  patches = []
  for H in range(0,3*sH,sH):
    for W in range(0,3*sW,sW):
      patches.append(inputs[:,:,H:H+sH,W:W+sW])
  
  X1 = []
  X2 = []
  y=[]

  for i in range(9):
    for j in range(9):
      X1.append(patches[i])
      X2.append(patches[j])
      y.append(torch.full((N,),j-i))

  X1 = torch.cat(X1,axis=0)
  X2 = torch.cat(X2,axis=0)
  y = torch.cat(y,axis=0)
  
  N = y.shape[0]
  prem = torch.randperm(N)

  return X1[prem],X2[prem],y[prem]


class Noise_Dataset(Dataset):
  def __init__(self,dset,output_size):
    self.dset = dset
    N = len(dset)
    self.y = torch.from_numpy(sample_spherical(N,output_size).transpose())

  def __getitem__(self,index):
    X = self.dset[index][0]
    box = self.dset[index][2]
    return (X,self.y[index],box)

  def __len__(self):
    return len(self.dset)