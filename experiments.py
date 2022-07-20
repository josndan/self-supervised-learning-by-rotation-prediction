from solver import *
from models import *
from supervision_transformation import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import sampler

INPUT_SIZE=128

def plot(train_acc,val_acc):
  # train_acc = [x.cpu() for x in train_acc]
  # val_acc = [x.cpu() for x in val_acc]

  plt.plot(train_acc, '-o')
  plt.plot(val_acc, '-o')
  plt.legend(['train', 'val'], loc='upper left')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.show()

def get_device_dtype():
  USE_GPU = True

  dtype = torch.float32 # we will be using float throughout this tutorial

  if USE_GPU and torch.cuda.is_available():
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')

  print('using device:', device)

  return device,dtype

def rotation_transform(loader,get_model,get_optim_sched,epoch =10,load_from=None):
  device,dtype = get_device_dtype()
  model = get_model(INPUT_SIZE,4,load_from=load_from)
  trainer = Solver(model,device,dtype,"./saved_models/rotation_upstream",loader)
  optimizer,scheduler = get_optim_sched(model)
  train_acc,val_acc = trainer.train(optimizer,scheduler,create_rotate_supervision,epochs=epoch)
  plot(train_acc,val_acc)
  return model

def context_transform(loader,get_model,get_optim_sched,epoch =10):
  device,dtype = get_device_dtype()
  model = get_model(INPUT_SIZE,4)
  trainer = Solver(model,device,dtype,"./saved_models/context_upstream",loader)
  optimizer,scheduler = get_optim_sched(model)
  train_acc,val_acc = trainer.train(optimizer,scheduler,create_rotate_supervision)
  plot(train_acc,val_acc)
  return model

def noise_transform(loader,get_model,get_optim_sched,epoch =10,output_size=10,load_from=None):
  device,dtype = get_device_dtype()
  model = get_model(INPUT_SIZE,output_size,load_from)
  trainer = Solver(model,device,dtype,"./saved_models/noise_upstream",to_noise_loader(loader,output_size))
  optimizer,scheduler = get_optim_sched(model)
  train_acc,val_acc = trainer.train(optimizer,scheduler,identity_transfrom,loss_type="l2",noise=True)
  plot(train_acc,val_acc)
  return model

def downstream_train(loader,num_classes,model,get_optim_sched,epoch =10):
  device,dtype = get_device_dtype()
  model = freeze(model)
  if model.save_name == "efficient_net":
    num_ftrs = model.classifier[1].in_features # = 1280
    model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
  elif model.save_name == "six_layer_net":
    num_ftrs = model[17].in_features #hidden_layer_size
    model[17] = nn.Linear(num_ftrs,num_classes)
  elif model.save_name == "mobile_net":
    pass

  train_acc,val_acc = trainer = Solver(model,device,dtype,"./saved_models/downstream",loader)
  optimizer,scheduler = get_optim_sched(model)
  trainer.train(optimizer,scheduler,identity_transfrom)
  plot(train_acc,val_acc)
  return model

def freeze(model):
  for param in model.parameters():
      param.requires_grad = False

  return model

def to_noise_loader(loader,output_size,num_train = 6000,batch_size=64):
  train_dset = Noise_Dataset(loader["train_dset"],output_size)
  val_dset = Noise_Dataset(loader["val_dset"],output_size)
  # test_dset = Noise_Dataset(loader["test_dset"],output_size)
  
  loader_train = DataLoader(train_dset, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(num_train)))


  loader_val = DataLoader(val_dset, batch_size=batch_size)

  # loader_test = DataLoader(test_dset, batch_size=batch_size)

  return {'train':loader_train, 'val': loader_val,"name":loader["name"]}#,'test':loader_test

