import os
import torch
import pickle
from tqdm import tqdm
from architectures import ResNet18_2
from sklearn.metrics import confusion_matrix
from data import make_dataset_shadows

def test_path(model, test_dataloader, save_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    misclassified_paths = []
    correct_paths = []

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_pred_probs = torch.tensor([]).to(device)
    # since we're not training, we don't need to calculate the gradients for our outputs
    
    with torch.no_grad():
        for paths, images, labels  in tqdm(test_dataloader, desc="testing"):
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)

            # calculate the probabilities
            probabilities = torch.nn.functional.softmax(outputs.data, 1)[:,1]

            correct_indices = ((probabilities < 0.5) & (labels == 0)) | ((probabilities > 0.5) & (labels == 1))
            correct_paths += [paths[idx] for idx, val in enumerate(correct_indices.cpu()) if val]

            misclassified_indices = ((probabilities > 0.5) & (labels == 0)) | ((probabilities < 0.5) & (labels == 1))
            misclassified_paths += [paths[idx] for idx, val in enumerate(misclassified_indices.cpu()) if val]

            # the class with the highest energy is what we choose as prediction
            predicted_labels = torch.argmax(outputs, dim = 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            all_predicted = torch.cat((all_predicted, predicted_labels))
            all_labels = torch.cat((all_labels, labels))
            all_pred_probs = torch.cat((all_pred_probs, outputs.data))

    # Let's say y_true is your true binary labels and y_pred_probs is the predicted probabilities for the positive class
    # You should replace them with your actual variables

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")
    tp = conf_matrix[0,0]
    fp = conf_matrix[1,0]
    fn = conf_matrix[0,1]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    accuracy = 100 * correct / total
    print(f"Accuracy for {save_path}: {accuracy}")

    print(f"{len(misclassified_paths) = }, {len(correct_paths) = }")
    with open('shadows/pickle/misclassified_shadow_outdoor.pkl', 'wb') as f:
        pickle.dump(misclassified_paths, f)
    
    with open('shadows/pickle/correct_shadow_outdoor.pkl', 'wb') as f:
        pickle.dump(correct_paths, f)

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