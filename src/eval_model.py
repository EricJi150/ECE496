import os
import yaml
import torch
import wandb
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from tqdm import tqdm
from data import make_dataset
import matplotlib.pyplot as plt
from architectures import ResNet18_5
from sklearn.metrics import confusion_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
    wandb.init(project='2D-FACT-Normalization-Tests')

    #import data
    test_loader = make_dataset.import_testsets()

    #create command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of config file")
    parser.add_argument("dataset", help="Name of dataset")
    args = parser.parse_args()
    print(args.config, args.dataset)

    #import model
    model = ResNet18_5().to(device)
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

def confusion():
    wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
    wandb.init(project='2D-FACT-Multi')
      
    #import data
    _, _, test_loader = make_dataset.import_train_multi()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of config file")
    parser.add_argument("model", help="Name of model")
    args = parser.parse_args()
    print(args.config, args.model)

    #import model
    model = ResNet18_5().to(device)
    save_path = os.path.join('../models','2D-FACT_'+args.config+'_'+args.model)
    model.load_state_dict(torch.load(save_path))

    #evaluate
    model.eval()
    true_labels = []
    pred_labels = []
    
    print("begin eval")
    for data, label in tqdm(test_loader):
      data, label = data.to(device), label.to(device)
      with torch.no_grad():
        output = model(data)
      pred = torch.argmax(output, dim=-1)
      true_labels.extend(label.cpu().numpy())
      pred_labels.extend(pred.cpu().numpy())
    
    #generate matrix
    matrix = confusion_matrix(true_labels, pred_labels)

    #visualize the matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    #save the matrix
    wandb.log({"Confusion Matrix"+args.config+'_'+args.model : wandb.Image(plt)})

if __name__ == "__main__":
    confusion()
