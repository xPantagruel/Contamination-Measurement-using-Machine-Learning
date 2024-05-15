from pathlib import Path
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from PIL import Image
from typing import Any, Callable, Optional
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import csv
import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision.datasets import VisionDataset

# In case we want to have more epochs, we can change the value here
num_epochs = 20
batch_size = 1

def create_segmentation_dataset(root, image_folder, mask_folder, transforms=None, subset=None, ImgColorMode="rgb", MskColorMode="grayscale"):
    # Validate paths and parameters
    fraction = 0.2
    seed = 42
    ImgFPath = Path(root) / image_folder
    MskFPath = Path(root) / mask_folder
    ValidColorM = {"rgb": "RGB", "grayscale": "L"}

    # Assign standardized color modes
    ImgColorMode = ValidColorM[ImgColorMode]
    MskColorMode = ValidColorM[MskColorMode]

    images = np.array(sorted(ImgFPath.glob("*")))
    masks = np.array(sorted(MskFPath.glob("*")))

    np.random.seed(seed)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    masks = masks[indices]

    if subset == "Test":
        split_index = int(np.ceil(len(images) * (1 - fraction)))
        Img_Names = images[split_index:]
        MaskNm = masks[split_index:]
    else:
        split_index = int(np.ceil(len(images) * (1 - fraction)))
        Img_Names = images[:split_index]
        MaskNm = masks[:split_index]

    # Utility function to get items
    def get_item(index):
        ImgPath = Img_Names[index]
        MaskPath = MaskNm[index]
        with open(ImgPath, "rb") as image_file, open(MaskPath, "rb") as mask_file:
            Img = Image.open(image_file).convert(ImgColorMode)
            Msk = Image.open(mask_file).convert(MskColorMode)
            Smpl = {"image": Img, "mask": Msk}
            if transforms:
                Smpl["image"] = transforms(Smpl["image"])
                Smpl["mask"] = transforms(Smpl["mask"])
            return Smpl

    return [get_item(i) for i in range(len(Img_Names))]

def training_Init(model):
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    return metrics, criterion, optimizer, best_model_wts, best_loss

def Get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_logger(bpath, metrics):
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames
        
def store_epoch_results(batchsummary, fieldnames, model, phase, loss, best_loss, best_model_wts, bpath):
    for field in fieldnames[3:]:
        batchsummary[field] = np.mean(batchsummary[field])
    print(batchsummary)
    with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(batchsummary)
        if loss < best_loss and phase == 'Test' :
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    return best_model_wts, best_loss

def log_results(epoch, train_loss, train_metrics, test_loss, test_metrics, fieldnames, bpath):
    # Prepare the dictionary to log
    log_entry = {
        'epoch': epoch,
        'Train_loss': train_loss,
        'Test_loss': test_loss
    }
    
    # Add training metrics to log_entry
    for metric_name, metric_value in train_metrics.items():
        log_entry[f'Train_{metric_name}'] = metric_value

    # Add testing metrics to log_entry
    for metric_name, metric_value in test_metrics.items():
        log_entry[f'Test_{metric_name}'] = metric_value

    # Writing to CSV file
    with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
        wrter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        wrter.writerow(log_entry)

    print(f"Logged Results for Epoch {epoch}: {log_entry}")

def train_epoch(model, LoadD, criterion, optimizer, metrics, device):
    model.train()
    running_loss = 0.0
    metric_scores = {name: [] for name in metrics.keys()}

    for Smpl in tqdm(LoadD):
        inputs = Smpl['image'].to(device)
        masks = Smpl['mask'].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        FLoss = criterion(outputs['out'], masks)
        FLoss.backward()
        optimizer.step()

        running_loss += FLoss.item() * inputs.size(0)
        PredY = outputs['out'].data.cpu().numpy().ravel()
        TrueY = masks.data.cpu().numpy().ravel()

        for name, metric in metrics.items():
            if name == 'auroc':
                metric_scores[name].append(metric(TrueY.astype('uint8'), PredY))
            else:
                metric_scores[name].append(metric(TrueY > 0, PredY > 0.1))

    epoch_loss = running_loss / len(LoadD.dataset)
    for metric in metric_scores:
        metric_scores[metric] = np.mean(metric_scores[metric])
    return epoch_loss, metric_scores

def test_epoch(model, dataloader, criterion, metrics, device):
    model.eval()
    LossR = 0.0
    metric_scores = {name: [] for name in metrics.keys()}

    with torch.no_grad():
        for sample in tqdm(dataloader):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs['out'], masks)
            LossR += loss.item() * inputs.size(0)
            PredY = outputs['out'].data.cpu().numpy().ravel()
            TrueY = masks.data.cpu().numpy().ravel()

            for name, metric in metrics.items():
                if name == 'auroc':
                    metric_scores[name].append(metric(TrueY.astype('uint8'), PredY))
                else:
                    metric_scores[name].append(metric(TrueY > 0, PredY > 0.1))

    epoch_loss = LossR / len(dataloader.dataset)
    for metric in metric_scores:
        metric_scores[metric] = np.mean(metric_scores[metric])
    return epoch_loss, metric_scores

def train_model(model, dataloaders, bpath):
    metrics, criterion, optimizer, bestWgths, best_loss = training_Init(model)
    device = Get_device()
    model.to(device)
    print(f"Using {device} for training")

    fieldnames = init_logger(bpath, metrics)

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        LossT, trainM = train_epoch(model, dataloaders['Train'], criterion, optimizer, metrics, device)
        LossTest, testM = test_epoch(model, dataloaders['Test'], criterion, metrics, device)

        if LossTest < best_loss:
            best_loss = LossTest
            bestWgths = copy.deepcopy(model.state_dict())

        log_results(epoch, LossT, trainM, LossTest, testM, fieldnames, bpath)

    model.load_state_dict(bestWgths)
    return model

def main():
    # this is the path to the dataset that has the images and masks and is in the same directory as the script
    data_directory = "Wholedataset"

    # this is the path to the directory where the training results will be stored
    exp_directory = "TrainingResults"
    data_directory = Path(data_directory)

    print (data_directory)
    
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    outChannels = 1
    model.classifier = DeepLabHead(2048, outChannels)
    model.train()

    data_transforms = transforms.Compose([transforms.ToTensor()])

    data = {
        x: create_segmentation_dataset(data_directory,
                               image_folder='Images',
                               mask_folder='Masks',
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    
    num_cores = os.cpu_count() - 1

    DataIn = {
        x: DataLoader(data[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_cores)
        for x in ['Train', 'Test']
    }
    
    print ("Training model")
    _ = train_model(model,
                    DataIn,
                    bpath=exp_directory)

    print("Saving model")
    torch.save(model, exp_directory / 'wghts.pt')

if __name__ == "__main__":
    main()