import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchmetrics.detection import MeanAveragePrecision

from torchvision.models import ResNet50_Weights
from torchvision.models.optical_flow.raft import ResidualBlock
from torchvision.transforms import v2

from VOC_Dataset import VOC
from create_CSV import generate_csv
from train_helper import show_examples, collate_fn, get_gpu_status, train_one_epoch, split_data, Eval_loss, eval_mAP

from validate_csv import validate_csv_files

num_classes = 1
rebuild_csv=False

print("******Preparing Envoirment******")
get_gpu_status()
device = torch.device('cpu')
print(torch.ones(1, device=device))


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("Device = " + str(device))
print("Random Seed = " + str(seed))

# split the data
print("******Preparing Data******")
transform = T.Compose([
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    # T.RandomHorizontalFlip(0.5),
    # T.RandomVerticalFlip(0.5),
    # T.RandomRotation(45),
])

if rebuild_csv:
    print("Rebuilding CSV Files")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/train/", "trainA.csv")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/test/", "testA.csv")
    generate_csv("/Users/olivereielson/Desktop/tennis-tracker/valid/", "validA.csv")


validate_csv_files(["trainA.csv", "testA.csv", "validA.csv"])

# define the dataset
train_data = VOC("trainA.csv", transform=transform)
valid_data = VOC("testA.csv", transform=transform)
test_data = VOC("validA.csv", transform=transform)

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("Train_loader Size = " + str(len(train_loader)))
print("Valid_loader Size = " + str(len(valid_loader)))
print("Test_loader Size = " + str(len(test_loader)))

print("******Preparing Model******")

epochs = 10
num_classes = 2
learning_rate = 0.01
step_size = 1
gamma = 0.1
backbone_weights = ResNet50_Weights

print("Backbone Weights = " + str(backbone_weights))
print("Learning Rate = " + str(learning_rate))
print("Step Size = " + str(step_size))
print("Gamma = " + str(gamma))
print("Number of Classes = " + str(num_classes))

model = torchvision.models.detection.retinanet_resnet50_fpn_v2(num_classes=num_classes,)
model.to(device)
# define the parameters


metric = MeanAveragePrecision(box_format='xyxy', iou_type="bbox")

metric.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, )

train_history = {"total_train_loss": [], "map": [], "eval_loss": []}

print(f"******Starting Training: {epochs} epochs******")
for i in range(epochs):
    # # train the model

    train_loss = train_one_epoch(model, optimizer, train_loader, device, lr_scheduler=lr_scheduler)
    train_history["total_train_loss"].append(train_loss)
    print(f"Epoch #{i} training loss: {train_loss}")

    eval_loss = Eval_loss(model, valid_loader, device)
    train_history["eval_loss"].append(eval_loss)
    print(f"Epoch #{i} eval loss: {eval_loss}")

    mAp = eval_mAP(model, valid_loader, device, metric)
    train_history["map"].append(mAp)
    print(f"Epoch #{i} mAp: {mAp}")

print("******Saving Model******")
torch.save(model, "fishing_model.sav")
