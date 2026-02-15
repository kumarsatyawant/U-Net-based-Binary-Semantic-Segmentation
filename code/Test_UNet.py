import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loaders.dataset_loader import get_test_loaders
from model.U_Net import U_Net
from Utils import load_checkpoint
import numpy as np
from PIL import Image


#Set Hyperparameter values
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13
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


PIXEL_ACC = []
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
    load_checkpoint(CHECKPOINT_DIR, model)

color_map = [
    (0,0,0),
    (70,70,70),     
    (190,153,153),     
    (250,170,150),
    (220,20,60),
    (153,153,153),
    (157,234,50),
    (128,64,128),
    (244,35,232),
    (107,142,35),
    (0,0,142),
    (102,102,156),
    (220,220,0),
]

colors = np.array(color_map, dtype=np.uint8)

def test_predictions(loader, model, colors, save_path=None, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y_1 = y.to('cpu').numpy()
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            preds = torch.argmax(softmax(model(x)),axis=1).to('cpu').numpy()
            print("preds shape: ", preds.shape)

            preds1 = np.array(preds[0,:,:])
            print("preds1 shape: ", preds1.shape)
            rgb_preds1 = colors[preds1]
            rgb_preds1 = Image.fromarray(rgb_preds1)
            rgb_preds1.save(os.path.join(save_path, 'Pred_RGB_1.png'))

            preds2 = np.array(preds[1,:,:])
            rgb_preds2 = colors[preds2]
            rgb_preds2 = Image.fromarray(rgb_preds2)
            rgb_preds2.save(os.path.join(save_path, 'Pred_RGB_2.png'))

            preds3 = np.array(preds[2,:,:])
            rgb_preds3 = colors[preds3]
            rgb_preds3 = Image.fromarray(rgb_preds3)
            rgb_preds3.save(os.path.join(save_path, 'Pred_RGB_3.png'))

            preds4 = np.array(preds[3,:,:])
            rgb_preds4 = colors[preds4]
            rgb_preds4 = Image.fromarray(rgb_preds4)
            rgb_preds4.save(os.path.join(save_path, 'Pred_RGB_4.png'))



            mask1 = np.array(y_1[0][:,:,0], dtype=np.uint8)
            print("mask1 shape: ", mask1.shape)
            rgb_mask1 = colors[mask1]
            rgb_mask1 = Image.fromarray(rgb_mask1)
            rgb_mask1.save(os.path.join(save_path, 'Mask_RGB_1.png'))

            mask2 = np.array(y_1[1][:,:,0], dtype=np.uint8)
            rgb_mask2 = colors[mask2]
            rgb_mask2 = Image.fromarray(rgb_mask2)
            rgb_mask2.save(os.path.join(save_path, 'Mask_RGB_2.png'))

            mask3 = np.array(y_1[2][:,:,0], dtype=np.uint8)
            rgb_mask3 = colors[mask3]
            rgb_mask3 = Image.fromarray(rgb_mask3)
            rgb_mask3.save(os.path.join(save_path, 'Mask_RGB_3.png'))

            mask4 = np.array(y_1[3][:,:,0], dtype=np.uint8)
            rgb_mask4 = colors[mask4]
            rgb_mask4 = Image.fromarray(rgb_mask4)
            rgb_mask4.save(os.path.join(save_path, 'Mask_RGB_4.png'))


check_accuracy(val_loader, model, device=DEVICE)

test_predictions(test_loader, model, colors, save_path=SAVE_TEST_PREDS_IMAGES)