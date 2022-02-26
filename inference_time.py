import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import torch.nn as nn
import time
from ptflops import get_model_complexity_info
from tqdm import tqdm

N = 5 # Number of inference time measurements post warmup epoch
MAX_BATCH_SIZE = 1024 # Batch size from 1 to MAX_BATCH_SIZE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Measurement on device {}.'.format(DEVICE))

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

def get_num_correct(preds,labels):
    return (preds.argmax(dim=1) == labels).sum().item()

def evaluate_model(model, data_loader):
    model.eval()
    total_inference_time = 0
    with torch.no_grad():
        for i, (img, lab) in enumerate(tqdm(data_loader)):
            img = img.to(DEVICE)
            start = time.time()
            y = model(img)
            end = time.time()
            total_inference_time += end-start
            del y
    print("Total Inference Time is: {}seconds.".format(total_inference_time))
    return total_inference_time

model = models.resnet50(pretrained=True)
# pruned_resnet_dict = torch.load('/home/tanay/Scripts/Inference/GReg_pruned_models/pruned_models/resnet50_2.31x_GReg-2_top1=75.36.pth', map_location=torch.device('cpu'))
# pruned_resnet = pruned_resnet_dict['model']
# pruned_resnet.load_state_dict(pruned_resnet_dict['state_dict'])
# model = pruned_resnet.module.to('cpu')

model = model.to(DEVICE)

print('Model Information...')
def ComputeFlops(net):
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

ComputeFlops(model)

BATCH_SIZE = 1
while BATCH_SIZE <= MAX_BATCH_SIZE:
    print("*** BATCH SIZE = {} ***".format(BATCH_SIZE))
    val_loader = data_loader(data_dir = imagenet_dir, batch_size = BATCH_SIZE, pin_memory = False)
    
    # Warmup
    print('Original Model...')
    print('Warmup...')
    evaluate_model(model, val_loader)
    print('Evaluating...')
    total_time = 0
    for i in range(N):
        print('Measurement iteration {}...'.format(i+1))
        current_time = evaluate_model(model, val_loader)
        total_time += current_time
    
    average_time = total_time/N
    print("Average Inference Time = {}".format(average_time))
    
    BATCH_SIZE = int(2*BATCH_SIZE)
