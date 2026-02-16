import os
import torch

def save_checkpoint(directory, state, filename='Checkpoint.tar.gz'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("=> Saving Checkpoint...")
    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=>Checkpoint Saved")

def load_checkpoint(directory, model, optimizer, filename='Checkpoint.tar.gz'):
    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> Loading Checkpoint...")
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        print("=> Checkpoint Loaded")
    else:
        print("Missing Checkpoint File")

def load_test_checkpoint(directory, model, filename='Checkpoint.tar.gz'):
    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> Loading Checkpoint...")
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Checkpoint Loaded")
    else:
        print("Missing Checkpoint File")