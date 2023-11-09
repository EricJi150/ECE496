import os
import yaml
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from architectures import ResNet18
from data import make_dataset

wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
wandb.init(project='2D-FACT-Normalization-Tests')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    #import data
    test_loader = make_dataset.import_testsets()

    #create command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of config file")
    parser.add_argument("dataset", help="Name of dataset")
    args = parser.parse_args()
    print(args.config, args.dataset)

    #import model
    model = ResNet18().to(device)
    save_path = os.path.join('../models','2D-FACT_'+args.config+'_'+args.dataset)
    model.load_state_dict(torch.load(save_path))
    
    #evalulate on each testset
    for idx in range(len(test_loader)):
        test_accuracy = eval(test_loader[idx], model)
        wandb.log({'Test accuracy': test_accuracy})

def eval(data_loader, model):
    model.eval()
    correct = 0
    total = 0

    it_test = tqdm(enumerate(data_loader), total=len(data_loader), desc="Validating ...", position = 1)
    for i, (images, labels) in it_test:
      images, labels = images.to(device), labels.to(device)
      with torch.no_grad():
        output = model(images)
      preds = torch.argmax(output, dim=-1)
      correct += (preds == labels).sum().item()
      total += len(labels)

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    main()
