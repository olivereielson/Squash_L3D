import colorsys
import math
import os
import random
import sys
import time
from torchmetrics.detection import MeanAveragePrecision
from sklearn.model_selection import train_test_split
import warnings
import tqdm
# Suppress all FutureWarnings
from torchvision import transforms
import torchvision
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
import torch.nn.functional as F
import matplotlib.pyplot as plt




def plot_training_history(data, save_path):

    assert data is not None, "No train data pased in"
    # Extract data
    epochs = range(1, len(data["total_train_loss"]) + 1)
    total_train_loss = data["total_train_loss"]
    eval_loss = data["eval_loss"]
    map_scores = data["map"]

    # Create the loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_train_loss, label="Total Train Loss", marker='o')
    plt.plot(epochs, eval_loss, label="Evaluation Loss", marker='o')
    plt.plot(epochs, map_scores, label="mAP", marker='o')
    plt.plot(epochs, data["map_50"], label="mAP_50", marker='o')
    plt.plot(epochs, data["mar_small"], label="mAR_small", marker='o')
    plt.plot(epochs, data["mar_1"], label="mar_1", marker='o')



    # Customize the plot
    plt.title("Training and Evaluation Loss and mAP over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the graph to the specified file
    plt.savefig(save_path+"/train_history.png")
    plt.close()  # Close the plot to free memory



def manage_checkpoints(check_point_dir, max_checkpoints=5):


    assert os.path.exists(check_point_dir), f"Checkpoint Dir does not exist at: {check_point_dir}"
    # Get a list of checkpoint files sorted by epoch number
    checkpoint_files = sorted(
        [f for f in os.listdir(check_point_dir) if f.startswith("checkpoint_epoch_")],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # Sort by epoch number
    )

    # Remove older checkpoints if exceeding `max_checkpoints`
    while len(checkpoint_files) > max_checkpoints:
        oldest_checkpoint = os.path.join(check_point_dir, checkpoint_files.pop(0))
        os.remove(oldest_checkpoint)
        print(f"\tOld Checkpoint Removed: {oldest_checkpoint}")



def load_checkpoints(checkpoint_dir):
    """
    Loads the latest checkpoint from the specified directory and returns the checkpoint dictionary.

    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.

    Returns:
        dict or None: The checkpoint dictionary if a checkpoint is found and loaded successfully; otherwise, None.
    """

    #make the dumb bitch shutup... if I get hack i get hacked
    warnings.filterwarnings("ignore", category=FutureWarning)



    if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
        # List all checkpoint files in the directory with .pth extension
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

        if files:
            # Find the latest checkpoint based on creation time
            latest_checkpoint = max(
                [os.path.join(checkpoint_dir, f) for f in files],
                key=os.path.getctime
            )
            print(f"Loading checkpoint from '{latest_checkpoint}'")

            try:
                checkpoint = torch.load(latest_checkpoint)
                print("Checkpoint loaded successfully.")
                return checkpoint
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return None
        else:
            print(f"No checkpoint files found in '{checkpoint_dir}'.")
            return None
    else:
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        os.mkdir(checkpoint_dir)
        assert os.path.exists(checkpoint_dir), "Failed to make checkpoint dir"
        return None



def get_gpu_status():
    if torch.backends.mps.is_available():
        if  torch.backends.mps.is_built():
            print("MPS is available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS is ready to be used")
    elif torch.cuda.is_available:
        print("Using CUDA GPU.... yaaa")
    else:
        print("Only the CPU is available")

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well. (From PyTorch documentation)
    """
    return tuple(zip(*batch))




def eval_mAP(model, dataloader, device, metric):
    model.eval()
    metric.reset()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            all_preds.extend([{'labels': p['labels'], 'boxes': p['boxes'], 'scores': p['scores']} for p in preds])
            all_targets.extend([{'labels': t['labels'], 'boxes': t['boxes']} for t in targets])

    metric.update(all_preds, all_targets)
    return metric.compute()



def Eval_loss(model, dataloader, device):
    total_loss = 0.0
    model.train() # need to take both image and tagets
    with torch.no_grad(): #we do not upade anything in th model
        for images, targets in dataloader:
            images = list(image.to(device,non_blocking=True) for image in images)
            targets = [{k: v.to(device,non_blocking=True) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            
            losses = sum(loss.item() for loss in loss_dict.values()) 
            total_loss += losses

    return total_loss / len(dataloader)


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler):

    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        # load data onto the GPU
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # train on data
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # check if I was messed something up
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return total_loss / len(data_loader)



def confidence_to_radar_color(confidence):
    """
    Map confidence (0 to 1) to a radar-style rainbow color using HSV.
    - 0 (low confidence): Light Blue
    - 1 (high confidence): Red
    """
    hue = (1 - confidence) * 0.67  # Map confidence to hue (red = 0, light blue ≈ 0.67)
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert HSV to RGB
    return int(r * 255), int(g * 255), int(b * 255)  # Scale to 0-255



def show_examples(model, dataloader, device,save_path, num_examples=5):
    """
    Show `num_examples` predictions from the dataset.
    """
    model.eval()
    number_shown = 0
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)

            for image, prediction, target in zip(images, predictions, targets):
                boxes = prediction["boxes"].tolist()
                boxes_real = target["boxes"].tolist()
                scores = prediction["scores"].tolist()

                t = transforms.ToPILImage()
                image = t(image)
                draw = ImageDraw.Draw(image)
                for box, score in zip(boxes, scores):
                    color = confidence_to_radar_color(score)  # Get RGB color from confidence
                    draw.rectangle(box, outline=color, width=3)

                for box in boxes_real:
                    try:
                        draw.rectangle(box, outline="pink", width=3)
                    except:
                        print("YOU FUCKED UP BIG TIME... THERE IS BAD DATA IN THE PIPELINE")
                        print(boxes_real)
                        print(f"Image number {number_shown}")

                #save the images
                filename = f"{number_shown}.png"
                filepath = os.path.join(save_path, filename)
                image.save(filepath)

                number_shown += 1
                if number_shown >= num_examples:
                    return




