from sys import flags
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

import pickle
import os.path

from custom_dataset import *

def get_cifar10_dataloaders(num_train,batch_size,input_size):
  transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#Cifar-10 mean and standard deviation
                T.Resize(input_size)
            ])

  cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                              transform=transform)
  loader_train = DataLoader(cifar10_train, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(num_train)))

  cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                            transform=transform)
  loader_val = DataLoader(cifar10_val, batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

  cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, 
                              transform=transform)
  loader_test = DataLoader(cifar10_test, batch_size=batch_size)

  return {'train':loader_train, 'val': loader_val,'test':loader_test,"name":"CIFAR_10"}


def get_COCO_dataloaders(num_train,batch_size,input_size,force_reload=False):

  if not force_reload and os.path.isfile("./dataset/COCO/pickled_with_dset.pickle") :
    with open("./dataset/COCO/pickled_with_dset.pickle", 'rb') as handle:
      return pickle.load(handle)

  transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), #Imagenet mean and standard deviation
                T.Resize((input_size,input_size))
            ])

  coco_train = COCO('./dataset/COCO', split="train", transform=transform)
  loader_train = DataLoader(coco_train, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(num_train)))

  coco_val = COCO('./dataset/COCO', split="validation", transform=transform)
  loader_val = DataLoader(coco_val, batch_size=batch_size)

  coco_test = COCO('./dataset/COCO', split="test", transform=transform)
  loader_test = DataLoader(coco_test, batch_size=batch_size)

  res = {'train':loader_train, 'val': loader_val,'test':loader_test,"name":"COCO","train_dset":coco_train
  ,"val_dset":coco_val,"test_dset":coco_test}

  with open('./dataset/COCO/pickled_with_dset.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return res

def get_citscape_dataloaders(num_train,batch_size,input_size,force_reload=False):

  if not force_reload and os.path.isfile("./dataset/cityscape/pickled.pickle") :
    with open("./dataset/cityscape/pickled.pickle", 'rb') as handle:
      return pickle.load(handle)

  transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), #Imagenet mean and standard deviation
                T.Resize((input_size,input_size))
            ])

  cityscape_train = cityscape('./dataset/cityscape', split="train", transform=transform)
  loader_train = DataLoader(cityscape_train, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(num_train)))

  cityscape_val = cityscape('./dataset/cityscape', split="val", transform=transform)
  loader_val = DataLoader(cityscape_val, batch_size=batch_size)

  cityscape_test = cityscape('./dataset/cityscape', split="test", transform=transform)
  loader_test = DataLoader(cityscape_test, batch_size=batch_size)

  res = {'train':loader_train, 'val': loader_val ,'test': loader_test,"name":"cityscape"}

  with open('./dataset/cityscape/pickled.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return res


def get_lfw_dataloaders(num_train,batch_size,input_size,force_reload=False):

  if not force_reload and os.path.isfile("./dataset/Labelled Faces in the wild/pickled.pickle") :
    with open("./dataset/Labelled Faces in the wild/new_pickled.pickle", 'rb') as handle:
      return pickle.load(handle)

  transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), #Imagenet mean and standard deviation
                T.Resize((input_size,input_size))
            ])

  lfw_train = lfw('./dataset/Labelled Faces in the wild/lfw', split="train", transform=transform)
  loader_train = DataLoader(lfw_train, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(num_train)))

  lfw_val = lfw('./dataset/Labelled Faces in the wild/lfw', split="val", transform=transform,paths = lfw_train.x_file_paths)
  loader_val = DataLoader(lfw_val, batch_size=batch_size)

  lfw_test = lfw('./dataset/Labelled Faces in the wild/lfw', split="test", transform=transform,paths = lfw_train.x_file_paths)
  loader_test = DataLoader(lfw_test, batch_size=batch_size)

  res = {'train':loader_train, 'val': loader_val ,'test': loader_test,"name":"lfw","train_dset":lfw_train
  ,"val_dset":lfw_val,"test_dset":lfw_test}

  with open('./dataset/Labelled Faces in the wild/new_pickled.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return res