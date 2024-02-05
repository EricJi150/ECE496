import os
import yaml
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from architectures import ResNet18_2
from architectures import ResNet50_2
from architectures import ResNet18_3_Multi
from data import make_dataset
from data import make_dataset_shadows

wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
wandb.init(project='2D-FACT-Multi')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    #create command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of config file")
    parser.add_argument("dataset", help="Name of dataset")
    # parser.add_argument("layers", help="Number of layers")
    parser.add_argument("channels", help="Number of channels")
    args = parser.parse_args()
    print(args.config, args.dataset, args.channels)

    #import data
    train_loader, val_loader, test_loader = make_dataset.import_train_multi()
    # if args.dataset == "indoor":
    #     train_loader, val_loader, test_loader = make_dataset_shadows.import_indoor_data()
    # elif args.dataset == "outdoor":
    #     train_loader, val_loader, test_loader = make_dataset_shadows.import_outdoor_data()

    #read config file
    config_file_name = args.config
    config_file_name += '.yml'
    with open(os.path.join('../configs/', config_file_name)) as file:
        config = yaml.safe_load(file)

    #set parameters from config
    num_epochs = config["num_epochs"]
    test_interval = config["test_interval"]
    learn_rate = config["learn_rate"]
    step_size_ = config["step_size"]
    gamma_ = config["gamma"]

    #setup model
    # if args.layers == "18":
    #     model =  ResNet18_2().to(device)
    # if args.layers == "50":
    #     model =  ResNet50_2().to(device)
    model = ResNet18_3_Multi
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_, gamma=gamma_)
    criterion = torch.nn.CrossEntropyLoss()

    #variables for early stopping
    curr_patience = 0
    patience = 3
    min_delta = 0.001
    best_val_accuracy = -np.inf

    #main training loop
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training ...", position=0):
        train_loss = train(train_loader, model, criterion, optimizer)
        lr_scheduler.step()
        wandb.log({'Epoch': epoch+1, 'Train loss': train_loss})
        val_accuracy = eval(val_loader, model)
        wandb.log({'Epoch': epoch+1, 'Val accuracy': val_accuracy})

        #save best model
        if (val_accuracy > best_val_accuracy + min_delta):
            save_path = os.path.join('../models/Multi','2D-FACT'+'_'+args.config+'_Multi'+args.layers)
            torch.save(model.state_dict(), save_path)
            print("saved best model")
            best_val_accuracy = val_accuracy
            curr_patience = 0
        else:
            curr_patience += 1

        #early stopping
        if (curr_patience == patience):
            break

    #test best model
    model.load_state_dict(torch.load(save_path))
    test_accuracy = eval(test_loader, model)
    wandb.log({'Test accuracy': test_accuracy})

    
    
def train(train_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0

    it_train = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training ...", position = 1)
    for i, (paths, images, labels) in it_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    return epoch_loss


def eval(data_loader, model):
    model.eval()
    correct = 0
    total = 0

    it_test = tqdm(enumerate(data_loader), total=len(data_loader), desc="Validating ...", position = 1)
    for i, (paths, images, labels) in it_test:
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
