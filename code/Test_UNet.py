import os
import torch
import torch.nn as nn
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2
from dataset_loaders.dataset_loader import get_test_loaders
from model.U_Net import U_Net
from Utils import load_test_checkpoint

#Set Hyperparameter values
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1
BATCH_SIZE = 8
NUM_WORKERS = 0
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = True


#Training, Validation and test image directory
VAL_IMG_DIR = '' # Specify the path
VAL_MASK_DIR = '' # Specify the path
TEST_IMG_DIR = '' # Specify the path
TEST_MASK_DIR = '' # Specify the path


#Checkpoint directory
CHECKPOINT_DIR = '' # Specify the path
SAVE_TEST_PREDS_IMAGES = '' # Specify the path


val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.4914, 0.4824, 0.4467],
            std=[0.2471, 0.2436, 0.2616],
        ),
        ToTensorV2()
    ]
)

test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.4914, 0.4824, 0.4467],
            std=[0.2471, 0.2436, 0.2616],
        ),
        ToTensorV2()
    ]
)


val_loader, test_loader = get_test_loaders(
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    BATCH_SIZE,
    val_transforms,
    test_transforms,
    NUM_WORKERS,
    PIN_MEMORY
)


val_features, val_labels = next(iter(val_loader))
print(f"Validation Feature batch shape: {val_features.size()}")
print(f"Validation Masks batch shape: {val_labels.size()}\n")

test_features, test_labels = next(iter(test_loader))
print(f"Test Feature batch shape: {test_features.size()}")
print(f"Test Labels batch shape: {test_labels.size()}\n")


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            temp_1 = torch.max(y, dim=3)[0]

            softmax = nn.Softmax(dim=1)

            preds_1 = softmax(model(x))
            preds = torch.argmax(preds_1,axis=1)
            num_correct += (preds == temp_1).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * temp_1).sum()) / ((preds + temp_1).sum())

    print(f"Validation Pixel Accuracy: {num_correct}/{num_pixels} | {num_correct/num_pixels*100:.2f}%")
    print(f"Validation Dice Score: {dice_score/len(loader)}")


model = U_Net(num_classes=NUM_CLASSES).to(DEVICE)


if LOAD_MODEL:
    load_test_checkpoint(CHECKPOINT_DIR, model)


def save_test_prediction(loader, model, save_directory, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        #Save Predicted Masks
        torchvision.utils.save_image(preds, f"{save_directory}/pred_{idx}.png")
        #Save original Ground truth Masks
        torchvision.utils.save_image(y.unsqueeze(1), f"{save_directory}/{idx}.png")


check_accuracy(val_loader, model, device=DEVICE)

save_test_prediction(test_loader, model, save_directory=SAVE_TEST_PREDS_IMAGES)