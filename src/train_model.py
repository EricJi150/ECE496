import os
import yaml
import torch
import wandb
import argparse
from tqdm import tqdm
from architectures import ResNet18
from data import make_dataset

wandb.login(key="76c1f7f13f849593c4dc0d5de21f718b76155fea")
wandb.init(project='2D-FACT')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    #import data
    train_loader, test_loader = make_dataset.import_data()

    #create command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of config file")
    args = parser.parse_args()

    #read config file and 
    with open(os.path.join('../configs/', args.config)) as file:
        config = yaml.safe_load(file)

    #set parameters from config
    num_epochs = config["num_epochs"]
    test_interval = config["test_interval"]
    learn_rate = config["learn_rate"]
    step_size_ = config["step_size"]
    gamma_ = config["gamma"]

    #setup model
    model =  ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_, gamma=gamma_)
    criterion = torch.nn.CrossEntropyLoss()

    #main training loop
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training ...", position=0):
        train_loss = train(train_loader, model, criterion, optimizer)
        lr_scheduler.step()
        wandb.log({'Epoch': epoch+1, 'Train loss': train_loss})

        if(epoch%test_interval==0 or epoch==num_epochs-1):
            train_accuracy = test(train_loader, model, criterion)
            wandb.log({'Epoch': epoch+1, 'Train accuracy': train_accuracy})
            test_accuracy = test(test_loader, model, criterion)
            wandb.log({'Epoch': epoch+1, 'Test accuracy': train_accuracy})
    
    #save model
    torch.save(model.state_dict(), os.path.join('../models','2D-FACT:'+args.config))

def train(train_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0

    it_train = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training ...", position = 1)
    for i, (images, labels) in it_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    return epoch_loss


def test(test_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0

    it_test = tqdm(enumerate(test_loader), total=len(test_loader), desc="Validating ...", position = 1)
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