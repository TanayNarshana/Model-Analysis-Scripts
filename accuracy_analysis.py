import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import torch.nn as nn
torch.manual_seed(17)
from tqdm import tqdm

# substitute location to your imagenet directory containing 'val' folder.
imagenet_dir = '/home/tanay/ImageNet/ILSVRC/Data/CLS-LOC'

BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(pretrained=True)
model = model.to(DEVICE)

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


def get_num_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def evaluate_model(model, data_loader):
    model.eval()
    total_correct_1, total_correct_5 = 0, 0
    total_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(data_loader)):
            images = images.to(DEVICE)
            target = target.to(DEVICE)
            output = model(images)
            correct_1, correct_5 = get_num_correct(output, target, topk=(1, 5))
            total_correct_1 += correct_1
            total_correct_5 += correct_5
            total_samples += target.shape[0]
            del output
    accuracy_1 = (100*total_correct_1)/total_samples
    accuracy_5 = (100*total_correct_5)/total_samples
    return accuracy_1, accuracy_5

val_loader = data_loader(data_dir = imagenet_dir, batch_size = BATCH_SIZE, pin_memory = False)
acc_1, acc_5 = evaluate_model(model, val_loader)
print('Top 1 Accuracy: {}%\nTop 5 Accuracy: {}%'.format(acc_1.item(), acc_5.item()))
