import torch
import os
import torch.nn.functional as F
import time
import numpy as np

class Solver:

  def __init__(self,model,device,dtype,file_name,data):
    self.model = model
    self.device = device
    self.dtype = dtype
    self.file_name = file_name + '/' + model.save_name
    self.data = data

  def check_accuracy(self,loader,supervision,dset_name):
    num_correct = 0
    num_samples = 0
    self.model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for instance in loader:
            if dset_name == 'COCO':
              x,y,boxes = instance
            else:
              x,y = instance
            x,y = supervision(x,y)
            x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            y = y.to(device=self.device, dtype= torch.long)#self.dtype)
            scores = self.model(x)
            
            _, preds = scores.max(1)
            
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc

  def train(self, optimizer,scheduler, supervision,epochs=10,print_every=100,loss_type="cross_entropy",noise=False):
    if loss_type == "cross_entropy":
      loss_fun = F.cross_entropy
    else :
      loss_fun = F.mse_loss
    
    extra = ""
    if "name" in self.data:
      extra = self.data["name"]
    filename = str(time.ctime(time.time())) +extra
    os.makedirs(os.path.dirname(self.file_name + "/" + "run"+str(filename)+".pth"), exist_ok=True)
    model = self.model.to(device=self.device)  # move the model parameters to CPU/GPU
    best_acc = float("-inf")
    min_loss = float("inf")
    # train_loss = []
    train_accs = []
    # val_loss = []
    val_accs = []
    for e in range(epochs):
        for t, instance in enumerate(self.data['train']):
            
            if self.data['name'] == 'COCO':
              x,y,boxes = instance
            else:
              x,y = instance

            x,y = supervision(x,y)
            
            model.train()  # put model to training mode

            x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            y = y.to(device=self.device, dtype=torch.long)#self.dtype)

            scores = model(x)
            
            loss = loss_fun(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print(e,' Iteration %d, loss = %.4f' % (t, loss.item()))
                # train_loss.append(loss.item())
                if noise and loss.item() < min_loss:
                  min_loss = loss.item()
                  torch.save(model.state_dict(), self.file_name + "/" + "run"+str(filename)+".pth")

                if not noise:
                  print("Train acc ")
                  train_acc = self.check_accuracy(self.data['train'],supervision,self.data['name'])
                  print("Validation acc ")
                  val_acc = self.check_accuracy(self.data['val'],supervision,self.data['name'])
                  train_accs.append(train_acc)
                  val_accs.append(val_acc)
                  if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), self.file_name + "/" + "run"+str(filename)+".pth")

                print()
        scheduler.step()
    
    
    return train_accs,val_accs


def check_accuracy_test(loader, model, supervision,device,dtype,dset_name):
    print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for instance in loader:
            if dset_name == 'COCO':
              x,y,boxes = instance
            else:
              x,y = instance
            x,y = supervision(x,y)
            # x,y = supervision(x,y)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc