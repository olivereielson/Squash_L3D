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
from VOC_Dataset import VOC, FlipWithBBoxes, ResizeWithBBoxes
from create_CSV import generate_csv
from train_helper import show_examples, collate_fn, get_gpu_status, train_one_epoch, Eval_loss, eval_mAP
from train_helper import *
from validate_csv import validate_csv_files
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import argparse
from datetime import datetime
from segment_transform import CourtTransform


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train FasterRCNN with hyperparameters passed from the terminal.")

    parser.add_argument("--job_id", type=str, default=None, help="Batch Job id")

    # Dataset paths with defaults
    parser.add_argument("--train_csv", type=str, default="/cluster/tufts/cs152l3dclass/oeiels01/train.csv",
                        help="Path to the training CSV file (default: ./train.csv)")
    parser.add_argument("--valid_csv", type=str, default="/cluster/tufts/cs152l3dclass/oeiels01/valid.csv",
                        help="Path to the validation CSV file (default: ./valid.csv)")
    parser.add_argument("--test_csv", type=str, default="/cluster/tufts/cs152l3dclass/oeiels01/test.csv",
                        help="Path to the test CSV file (default: ./test.csv)")
    parser.add_argument("--data_dir", type=str, default="/cluster/tufts/cs152l3dclass/oeiels01/upload_dataset",
                        help="Directory containing the dataset (default: ./dataset)")
    parser.add_argument("--check_point_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: ./checkpoints)")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train (default: 25)")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate (default: 0.005)")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for learning rate scheduler (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay for optimizer (default: 0.0009)")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training (default: 32)")
    parser.add_argument("--valid_batch_size", type=int, default=16, help="Batch size for validation (default: 16)")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (including background) (default: 2)")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--show_outputs", action="store_true", help="Show example outputs")
    parser.add_argument("--checkpoints", action="store_true", help="Enable Checkpoint Saving")

    return parser.parse_args()


# Helper function for conditional printing
def log(message, verbose=True):
    if verbose:
        print(message)


# Main training function
def main(args):
    log("******Preparing Environment******", args.verbose)
    if args.verbose:
        get_gpu_status()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(torch.ones(1, device=device), args.verbose)

    # Set random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log(f"Device: {device}", args.verbose)
    log(f"Random Seed: {seed}", args.verbose)

    # Data preparation
    log("******Preparing Data******", args.verbose)
    transform = v2.Compose([
        CourtTransform(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWithBBoxes(),
        FlipWithBBoxes(flip_type="vertical", probability=0.35),
        FlipWithBBoxes(flip_type="horizontal", probability=0.35)
    ])

    train_data = VOC(args.train_csv, transform=transform, data_dir=args.data_dir)
    valid_data = VOC(args.valid_csv, transform=transform, data_dir=args.data_dir)
    test_data = VOC(args.test_csv, transform=transform, data_dir=args.data_dir)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                               collate_fn=collate_fn, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=True,
                                               collate_fn=collate_fn, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True,
                                              collate_fn=collate_fn, num_workers=2)

    log(f"Train_loader Size = {len(train_loader)}, {len(train_data.removed_images)} images removed", args.verbose)
    log(f"Valid_loader Size = {len(valid_loader)}, {len(valid_data.removed_images)} images removed", args.verbose)
    log(f"Test_loader Size = {len(test_loader)}, {len(test_data.removed_images)} images removed", args.verbose)

    # Model setup
    log("******Preparing Model******", args.verbose)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    model.to(device)

    # metric = MeanAveragePrecision(box_format='xyxy', iou_type="bbox", iou_thresholds=[0.2, 0.5, 0.75])
    metric = MeanAveragePrecision(box_format='xyxy', iou_type="bbox",class_metrics=False,backend="faster_coco_eval")
    metric.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_history = {
        "total_train_loss": [],
        "map": [],
        "eval_loss": [],
        "hyperparams": vars(args),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': -1,
        'job_id': args.job_id
    }

    log("******Restoring Checkpoints******", args.verbose)
    if args.checkpoints:
        checkpoint = load_checkpoints(args.check_point_dir)
        if checkpoint is not None:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                train_history = checkpoint['train_history']
                log(f"Restored from last checkpoint...Training from epoch {len(train_history["total_train_loss"])}",
                    args.verbose)
            except:
                print(f"Failed to load from checkpoint...you fucked up")

    # Training loop
    log("******Starting Training******", args.verbose)
    for epoch in tqdm.tqdm(range(len(train_history["total_train_loss"]), args.epochs), desc="Training Progress",
                           unit="epoch"):

        args.verbose and tqdm.tqdm.write(f"Epoch {epoch}")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, lr_scheduler=lr_scheduler)
        train_history["total_train_loss"].append(float(train_loss))
        args.verbose and tqdm.tqdm.write(f"\tTraining Loss: {train_loss}")

        eval_loss = Eval_loss(model, valid_loader, device)
        train_history["eval_loss"].append(float(eval_loss))
        args.verbose and tqdm.tqdm.write(f"\tValidation Loss: {eval_loss}")

        mAP = eval_mAP(model, valid_loader, device, metric)
        for key, value in mAP.items():
            if key not in train_history:
                train_history[key] = []
            train_history[key].append(float(value))

        args.verbose and tqdm.tqdm.write(f"\tmAP_50: {mAP["map_50"]}")

        mAP_train = eval_mAP(model, train_loader, device, metric)

        for key, value in mAP_train.items():
            key = f"{key}_train"
            if key not in train_history:
                train_history[key] = []
            train_history[key].append(float(value))

        args.verbose and tqdm.tqdm.write(f"\tmAP_50_train: {mAP_train["map_50"]}")

        with open(os.path.join(args.check_point_dir, "train_history.json"), "w") as f:
            json.dump(train_history, f)

        # if args.checkpoints and (epoch + 1) % 2 == 0:
        if args.checkpoints:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "train_history": train_history,
            }
            checkpoint_path = os.path.join(args.check_point_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_path)
            manage_checkpoints(args.check_point_dir, 2)
            args.verbose and tqdm.tqdm.write(f"\tModel checkpoint saved: {checkpoint_path}")

        if args.show_outputs:
            # save some examples to santity check the work
            # show_examples(model, test_loader, device, "Examples", num_examples=5)
            show_examples(model, valid_loader, device, "Examples", num_examples=10)
            show_examples(model, train_loader, device, "Examples2", num_examples=5)
            # Plot the training history
            plot_training_history(train_history, "Examples")

    # Save final model and training history
    log("******Saving Final Model******", args.verbose)
    train_history["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(args.check_point_dir, exist_ok=True)
    torch.save(model, os.path.join(args.check_point_dir, "final_model.sav"))
    with open(os.path.join(args.check_point_dir, "train_history.json"), "w") as f:
        json.dump(train_history, f)

    # save to folder of just past runs
    if args.job_id is not None:
        os.makedirs(f"{args.job_id}", exist_ok=True)
        with open(f"{args.job_id}/train_history.json", "w") as f:
            json.dump(train_history, f)

        # save training history
        plot_training_history(train_history, f"{args.job_id}")

    log("******Training Complete******", args.verbose)


if __name__ == "__main__":
    args = parse_args()
    main(args)