import torch
import matplotlib.pyplot as plt
import random
from utils  import *

import json
import matplotlib.patches as patches

def compute_saliency_maps(X, y, model,threshold):
    model.eval()
    X.requires_grad_()
    
    saliency = None
    score = model(X)
    loss = torch.sum(score.gather(1, y.view(-1, 1)).squeeze())
    # loss = torch.mean(score)

    loss.backward()
    saliency,_ = torch.max(torch.abs(X.grad),dim=1)
    X.grad = None
    return normalize_saliency(saliency,threshold)

def show_saliency_maps(model, X, y,class_names,threshold,boxes=None,dis=True):
    X_tensor = X
    y_tensor = torch.LongTensor(y)

    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)
    
    saliency = compute_saliency_maps(X_tensor, y_tensor, model,threshold)

    # plt.figure(dpi=1200)
    X = unnormalize(X)
    saliency = saliency.cpu().detach().numpy()
    X = X.detach().numpy()
    X = X.transpose((0,2,3,1))
    N = X.shape[0]
    # y = y.tolist()
    if dis :
      for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        # plt.title(class_names[y[i].item()])

        # ax = plt.gca()
        # x_img,y_img,w,h = json.loads(boxes[i])[0]
        # rect = patches.Rectangle((x_img, y_img), h, w, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)

        # ax = plt.gca()
        # x_img,y_img,w,h = json.loads(boxes[i])[0]
        # rect = patches.Rectangle((x_img, y_img), h, w, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
      plt.show()

    return saliency

def normalize_saliency(saliency,threshold = 0.5):
    res = torch.zeros_like(saliency)

    res[saliency > threshold] = 1

    return res

def show_saliency_maps_noise(model, X, y,class_names):
    X_tensor = X
    # y_tensor = torch.LongTensor(y)

    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    X_tensor = X_tensor.to(device)
    # y_tensor = y_tensor.to(device)
    
    saliency_maps = []
    for i in range(y.shape[1]):
      y = torch.tensor([i]*X.shape[0])
      y_tensor = y
      saliency_maps.append(compute_saliency_maps(X_tensor, y_tensor, model))
    res = torch.stack(saliency_maps,axis=0)
    saliency = res.mean(axis =0)
    # plt.figure(dpi=1200)
    X = unnormalize(X)
    saliency = saliency.cpu().detach().numpy()
    X = X.detach().numpy()
    X = X.transpose((0,2,3,1))
    N = X.shape[0]
    # y = y.tolist()
    for i in range(N):
      plt.subplot(2, N, i + 1)
      plt.imshow(X[i])
      plt.axis('off')
      # plt.title(class_names[y[i].item()])
      plt.subplot(2, N, N + i + 1)
      plt.imshow(saliency[i], cmap=plt.cm.hot)
      plt.axis('off')
      plt.gcf().set_size_inches(12, 5)
    plt.show()

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.
    
    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes
    
    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X

def create_class_visualization(target_y, model, dtype,class_names, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
    img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()
    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    img = img.to(device)

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        ########################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term!                                              #
        # Be very careful about the signs of elements in your code.            #
        ########################################################################
        scores = torch.squeeze(model(img))
        loss = scores[target_y] - l2_reg * (torch.linalg.vector_norm(img) **2)
        loss.backward()

        with torch.no_grad():
          dimg = learning_rate * img.grad
          img += dimg
          img.grad.data.zero_()
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        
        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-IMAGENET_MEAN[c] / IMAGENET_STD[c])
            hi = float((1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)
        
        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess(img.data.clone().cpu()))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.show()

