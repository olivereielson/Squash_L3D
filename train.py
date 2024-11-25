import pandas as pd
import torch
import torchvision
import tqdm
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchmetrics.detection import MeanAveragePrecision
import time
import os
import json
from torchvision.models import ResNet50_Weights
from torchvision.models.optical_flow.raft import ResidualBlock
from torchvision.transforms import v2
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from VOC_Dataset import VOC
from create_CSV import generate_csv
from train_helper import show_examples, collate_fn, get_gpu_status, train_one_epoch, split_data, Eval_loss, eval_mAP
from train_helper import *
from validate_csv import validate_csv_files
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

num_classes = 1
rebuild_csv=False

print("******Preparing Envoirment******")
get_gpu_status()
device = torch.device('cuda')
print(torch.ones(1, device=device))

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


check_point_dir = "/cluster/tufts/cs152l3dclass/oeiels01/synthetic"


print("Device = " + str(device))
print("Random Seed = " + str(seed))

# split the data
print("******Preparing Data******")



transform = v2.Compose([
    #DONT USE RESIZE IT DOES NOT DO LABELS
    # v2.Resize((640, 640)),  # Resizes image and adjusts bounding boxes
    v2.ToImage(),  # Ensures the input is a PIL Image
    v2.ToDtype(torch.float32, scale=True),  # Converts image to float32 and scales
    # Include other transforms as needed
])


# define the dataset
train_csv = "/cluster/tufts/cs152l3dclass/oeiels01/train+synthetic.csv"
test_csv = "/cluster/tufts/cs152l3dclass/oeiels01/test.csv"
valid_csv = "/cluster/tufts/cs152l3dclass/oeiels01/valid.csv"

train_data = VOC(train_csv, transform=transform,data_dir="/cluster/tufts/cs152l3dclass/oeiels01/upload_dataset")
valid_data = VOC(valid_csv, transform=transform,data_dir="/cluster/tufts/cs152l3dclass/oeiels01/upload_dataset")
test_data = VOC(test_csv, transform=transform,data_dir="/cluster/tufts/cs152l3dclass/oeiels01/upload_dataset")

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn,num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=True, collate_fn=collate_fn,num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, collate_fn=collate_fn,num_workers=2)

print("Train_loader Size = " + str(len(train_loader)))
print("Valid_loader Size = " + str(len(valid_loader)))
print("Test_loader Size = " + str(len(test_loader)))

print("******Preparing Model******")

epochs = 25
num_classes = 2
learning_rate = 1
step_size = 5
gamma = 0.05
weight_decay= 0.0005

weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT

print("Backbone Weights = " + str(weights))
print("Learning Rate = " + str(learning_rate))
print("Step Size = " + str(step_size))
print("Gamma = " + str(gamma))
print("Number of Classes = " + str(num_classes))

# model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=weights,
#                                                                num_classes=num_classes,    
#                                                                pretrained_backbone=False, ) 


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes,)


model.to(device)


metric = MeanAveragePrecision(box_format='xyxy', iou_type="bbox",iou_thresholds=[0.2,0.5,0.75])

metric.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,)


print("******Restoring Checkpoints******")
checkpoint = load_checkpoints(check_point_dir)
start_epoch = 0
if checkpoint is not None:
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_history = checkpoint['train_history']
        print(f"Restored from last checkpoint...Training from epoch {start_epoch}")
    except:
        print(f"Failed to load from checkpoint...you fucked up")


train_history = {"total_train_loss": [], "map": [], "eval_loss": []}

device = next(model.parameters()).device
print(f"The model is on device: {device}")

print(f"******Starting Training: {epochs} epochs******")
for i in tqdm.tqdm(range(start_epoch,epochs), desc="Training Progress", unit="epoch"):

    start_time = time.time()
    tqdm.tqdm.write(f"Epoch #{i}")



    #train the model
    train_loss = train_one_epoch(model, optimizer, train_loader, device, lr_scheduler=lr_scheduler)
    train_history["total_train_loss"].append(float(train_loss))
    tqdm.tqdm.write(f"\ttraining loss: {train_loss}")

    #do eval loss
    eval_loss = Eval_loss(model, valid_loader, device)
    train_history["eval_loss"].append(float(eval_loss))
    tqdm.tqdm.write(f"\teval loss: {eval_loss}")

    #do MAP
    mAp = eval_mAP(model, valid_loader, device, metric)
    train_history["map"].append(float(mAp))
    tqdm.tqdm.write(f"\tmAP: {mAp}")

    # Save the model state every 2 epochs
    if i % 2 ==0:
        checkpoint = {
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_history': train_history,
         }
        checkpoint_path = os.path.join(check_point_dir, f'checkpoint_epoch_{i}.pth')
        torch.save(checkpoint, checkpoint_path)
        tqdm.tqdm.write(f"\tModel Checkpoint Saved")




    #save some examples to santity check the work
    show_examples(model,test_loader,device,"Examples",num_examples=10)


    epoch_time = time.time() - start_time
    tqdm.tqdm.write(f"\tThis took {epoch_time:.2f} seconds")
    
    #save the train history
    with open(f'{check_point_dir}/train_history.json', 'w') as f:
        json.dump(train_history, f)



print("******Saving Model******")

torch.save(model, f"{check_point_dir}/model.sav")



print("ALL DONE")