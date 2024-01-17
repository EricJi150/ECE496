import os
import torch
import pickle
from tqdm import tqdm
from architectures import ResNet18_2
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data import make_dataset_shadows

def test_path(model, test_dataloader, save_path):
    margin = 0.445

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    misclassified_paths = []
    unconfident_paths = []

    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    # since we're not training, we don't need to calculate the gradients for our outputs
    
    with torch.no_grad():
        for paths, images, labels  in tqdm(test_dataloader, desc="testing"):
        
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)

            # calculate the probabilities
            probabilities = torch.nn.functional.softmax(outputs.data, 1)[:,1]

            unconfident_indices_real = (probabilities > 0.5) & (probabilities < 0.5 + margin) & (labels == 1)
            unconfident_indices_gen = (probabilities > 0.5) & (probabilities < 0.5 + margin) & (labels == 1)
            unconfident_paths += list(paths[unconfident_indices_real.cpu()])
            unconfident_paths += list(paths[unconfident_indices_gen.cpu()])

            misclassified_indices = ((probabilities > 0.5) & (labels == 0)) | ((probabilities < 0.5) & (labels == 1))
            misclassified_paths += [paths[idx] for idx, val in enumerate(misclassified_indices.cpu()) if val]

            # the class with the highest energy is what we choose as prediction
            predicted_labels = torch.argmax(outputs, dim = 1)
            all_predicted = torch.cat((all_predicted, predicted_labels))
            all_labels = torch.cat((all_labels, labels))

    # Let's say y_true is your true binary labels and y_pred_probs is the predicted probabilities for the positive class
    # You should replace them with your actual variables

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")
    tp = conf_matrix[0,0]
    tn = conf_matrix[1,1]
    fp = conf_matrix[1,0]
    fn = conf_matrix[0,1]
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


    print(f"{len(misclassified_paths) = }, {len(unconfident_paths) = }")
    with open('shadows/pickle/misclassified_shadow_outdoor.pkl', 'wb') as f:
        pickle.dump(misclassified_paths, f)
    with open('shadows/pickle/unconfident_shadow_outdoor.pkl', 'wb') as f:
        pickle.dump(unconfident_paths, f)

    unconfident_all_labels = torch.cat((unconfident_all_labels, labels[unconfident_indices_real]))
    unconfident_all_pred_probs = torch.cat((unconfident_all_pred_probs, outputs[unconfident_indices_real].data))

    fpr, tpr, thresholds = roc_curve(all_labels.cpu(), unconfident_all_pred_probs.cpu(), unconfident_pos_label = 1)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Unconfident Test Set ')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig('shadows/roc/misclassified_outdoor',dpi=200)
    

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model =  ResNet18_2().to(device)
    save_path = os.path.join('../models','Shadows'+'two'+'_'+'outdoor')
    model.load_state_dict(torch.load(save_path))
    test_loader = make_dataset_shadows.import_outdoor_data()
    test_path(model, test_loader, 'outdoor')

if __name__ == "__main__":
    main()