import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loaders.dataset_loader import get_loaders
from model.U_Net import U_Net
from Utils import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt

#Set Hyperparameter values
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 30
NUM_WORKERS = 0
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
NUM_CLASS = 1

#Training and Validation image directory
TRAIN_IMG_DIR = '' # Specify the path
TRAIN_MASK_DIR = '' # Specify the path
VAL_IMG_DIR = '' # Specify the path
VAL_MASK_DIR = '' # Specify the path


CHECKPOINT_DIR = '' # Specify the path
SAVE_TRAINING_CURVES = '' # Specify the path

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.4914, 0.4824, 0.4467],
            std=[0.2471, 0.2436, 0.2616],
        ),
        ToTensorV2()
    ]
)

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


train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    BATCH_SIZE,
    train_transform,
    val_transforms,
    NUM_WORKERS,
    PIN_MEMORY
)


train_features, train_labels = next(iter(train_loader))
print(f"Train Feature batch shape: {train_features.size()}")
print(f"Train Masks batch shape: {train_labels.size()}\n")

val_features, val_labels = next(iter(val_loader))
print(f"Validation Feature batch shape: {val_features.size()}")
print(f"Validation Masks batch shape: {val_labels.size()}")


TRAIN_LOSSES = []
def train(epoch, loader, model, optimizer, loss_fn):
    model.train()
    train_loss = 0

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # print(targets.shape)

        optimizer.zero_grad()

        # forward
        predictions = model(data)
        # print(predictions.shape)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f})'.format(epoch, batch_idx, train_loss / (batch_idx + 1)))

    TRAIN_LOSSES.append(train_loss / (batch_idx + 1))


DICE_COEF = []
PIXEL_ACC = []
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum())

    print(f"Validation Pixel Accuracy: {num_correct}/{num_pixels} | {num_correct/num_pixels*100:.2f}%")
    print(f"Validation Dice Score: {dice_score/len(loader)}")
    DICE_COEF.append((dice_score/len(loader))*100)
    PIXEL_ACC.append((num_correct/num_pixels)*100)

    return ((num_correct/num_pixels)*100)


model = U_Net(num_class=NUM_CLASS).to(DEVICE)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=1e-4, nesterov=True)


def main():
    #Load checkpoints if saved and LOAD_MODEL = "True"
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_DIR, model, optimizer)

    best_acc = 0
    for epoch in range(1, (NUM_EPOCHS+1)):
        train(epoch, train_loader, model, optimizer, loss_fn)

        #Check Validation Accuracy
        val_acc = check_accuracy(val_loader, model, device=DEVICE)

        if (val_acc>best_acc):
            best_acc = val_acc

            #Save model checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': epoch
            }

            save_checkpoint(CHECKPOINT_DIR, checkpoint)


    plt.figure(figsize=(10,5))
    plt.title("Training Loss")
    plt.plot(TRAIN_LOSSES,label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_TRAINING_CURVES, 'Train_Loss'), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title(" Validation Pixel Accuracy")
    plt.plot(PIXEL_ACC,label="Pixel Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Pixel Accuracy")
    plt.legend()
    plt.savefig(os.path.join(SAVE_TRAINING_CURVES, 'Val_Pixel_Acc'), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title(" Validation Dice Coefficient")
    plt.plot(DICE_COEF,label="Dice score")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.savefig(os.path.join(SAVE_TRAINING_CURVES, 'Val_Dice_Coeff'), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    main()