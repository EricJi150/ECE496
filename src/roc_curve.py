import pickle
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
from architectures import ResNet18_2
from data import make_dataset



def full_test(model, test_dataloader, mode = "Full", save_to_file = None, title):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_generated_probs = torch.tensor([]).to(device)
    # since we're not training, we don't need to calculate the gradients for our outputs

    
    with torch.no_grad():
        for _, images, labels in tqdm(test_dataloader, desc="testing"):
            images = images.float().to(device)
            labels = labels.to(device)
        
            predictions = model(images)
            
            probabilities = nn.Softmax(dim = 1)(predictions)
            generated_probabilities = probabilities[:, 1]

            predicted_labels = torch.argmax(predictions, dim = 1)
            
            total += labels.size(0)
            
            correct += (predicted_labels == labels).sum().item()
            
            all_predicted = torch.cat((all_predicted, predicted_labels))
            
            all_labels = torch.cat((all_labels, labels))
            
            all_generated_probs = torch.cat((all_generated_probs, generated_probabilities))
            
    # conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    # if mode == "Misclassified":
    #     print(f"Only Misclassified Test Set Confusion Matrix for:")
    # elif mode == "Easy":
    #     print("Easy (No Misclassified or Unconfident) Confusion Matrix:")
    # elif mode == "Unconfident":
    #     print("Only Unconfident Test Set Confusion Matrix:")
    # elif mode == "Misclassified and Unconfident":
    #     print("Both Misclassified and Unconfident Test Set Confusion Matrix:")
    # elif mode == "Full":
    #     print("Full Test Set Confusion Matrix:")
    # print(conf_matrix)
    # print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
    # tn = conf_matrix[1,1]
    # tp = conf_matrix[0,0]
    # fp = conf_matrix[1,0]
    # fn = conf_matrix[0,1]
    # print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    # print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    # accuracy = 100 * correct / total
    # print(f"accuracy: {accuracy}")
    
    fpr, tpr, thresholds = roc_curve(all_labels.cpu(), all_generated_probs.cpu(), pos_label = 1)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    if save_to_file is not None:
        print("saving figure")
        fig.savefig(save_to_file,dpi=200)
        
    # with open(f'shadows/pickle/outdoor_two_{mode}.pkl', 'wb') as f:
    #     pickle.dump([fpr, tpr, roc_auc], f)
    return

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model =  ResNet18_2().to(device)
    save_path = os.path.join('../models','Shadows'+'two'+'_'+'outdoor')
    model.load_state_dict(torch.load(save_path))
    # train_loader, val_loader, test_loader = make_dataset.import_indoor_data()
    train_loader, val_loader, test_loader = make_dataset.import_outdoor_data()
    full_test(model, test_loader,"Full", 'outdoor_two')


if __name__ == "__main__":
    main()