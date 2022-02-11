#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import torch.nn as nn
torch.manual_seed(17)
import time

from ptflops import get_model_complexity_info


# In[2]:


imagenet_dir = '/home/tanay/ImageNet/ILSVRC/Data/CLS-LOC'


def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  
def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset

def data_loader(data_dir, batch_size=256, workers=1, pin_memory=True):
    val_ds = val_dataset(data_dir)
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    return val_loader


# In[3]:


def get_num_correct(preds,labels):
    return (preds.argmax(dim=1) == labels).sum().item()

def evaluate_model(model, data_loader, progress_steps = 50, FP16 = False, limit = 9999999):
    model.eval()
    count = 0
    correct = 0
    total = 0
    total_inference_time = 0
    with torch.no_grad():
        for img, lab in data_loader:
            count += 1
            if FP16:
                img = img.half()
            start = time.time()
            y = model(img)
            end = time.time()
            correct += get_num_correct(y, lab)
            total += lab.shape[0]
            total_inference_time += end-start
            del y
            if count == limit:
                break
            # if count % progress_steps == 0:
            #    print("{} Batches evaluated.".format(count))
    accuracy = (100*correct)/total
    print("Total Inference Time is: {}seconds.".format(total_inference_time))
    if limit == 9999999:
        print("Accuracy is: {}%.".format(accuracy))
    return accuracy, total_inference_time


# In[4]:


resnet = models.resnet50(pretrained=True)
pruned_resnet_dict = torch.load('/home/tanay/Scripts/Inference/GReg_pruned_models/pruned_models/resnet50_2.31x_GReg-2_top1=75.36.pth', map_location=torch.device('cpu'))
pruned_resnet = pruned_resnet_dict['model']
pruned_resnet.load_state_dict(pruned_resnet_dict['state_dict'])
pruned_resnet = pruned_resnet.module.to('cpu')


# In[ ]:
LIMIT = 9999999

N = 5

BATCH_SIZE = 1
while BATCH_SIZE <= 256:
    print("*** BATCH SIZE = {} ***".format(BATCH_SIZE))
    val_loader = data_loader(data_dir = imagenet_dir, batch_size = BATCH_SIZE, pin_memory = False)
    
    # Warmup
    print('Original Model...')
    print('Warmup...')
    evaluate_model(resnet, val_loader, limit = LIMIT)
    print('Evaluating...')
    total_time = 0
    for i in range(N):
        _, current_time = evaluate_model(resnet, val_loader, limit = LIMIT)
        total_time += current_time
    
    unpruned_time = total_time/N
    print("Average Unpruned Inference Time = {}".format(unpruned_time))
    
    print('Pruned Model')
    print('Warmup...')
    evaluate_model(pruned_resnet, val_loader, limit = LIMIT)
    print('Evaluating...')
    total_time = 0
    for i in range(N):
        _, current_time = evaluate_model(pruned_resnet, val_loader, limit = LIMIT)
        total_time += current_time
        
    pruned_time = total_time/N
    print("Average Pruned Inference Time = {}".format(pruned_time))
    
    print('*** BATCH SIZE = {}, SPEEDUP = {} ***'.format(BATCH_SIZE, unpruned_time/pruned_time))
    
    BATCH_SIZE = int(2*BATCH_SIZE)


# In[ ]:

print('Accuracies...')
print('Pruned Model')
evaluate_model(pruned_resnet, val_loader)


print('Model Parameters for Pruned Model...')
def ComputeFlops(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

ComputeFlops(pruned_resnet)

