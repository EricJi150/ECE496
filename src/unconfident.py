import os
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
from architectures import ResNet18_2
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import make_dataset_shadows

def test_path(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    misclassified_paths = []
    unconfident_paths = []

    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for paths, images, labels  in tqdm(test_dataloader, desc="testing"):
        
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.nn.Softmax(dim = 1)(outputs.data)

            margin = 0.435

            # unconfident_indices_real = (probabilities > 0.5) & (probabilities < 0.5 + margin) & (labels == 1)
            # unconfident_indices_gen = (probabilities < 0.5) & (probabilities > 0.5 - margin) & (labels == 0)
            # unconfident_paths += [paths[idx] for idx, val in enumerate(unconfident_indices_real.cpu()) if val]
            # unconfident_paths += [paths[idx] for idx, val in enumerate(unconfident_indices_gen.cpu()) if val]

            # misclassified_indices = ((probabilities > 0.5) & (labels == 0)) | ((probabilities < 0.5) & (labels == 1))
            # misclassified_paths += [paths[idx] for idx, val in enumerate(misclassified_indices.cpu()) if val]

            predicted_labels = torch.argmax(outputs, dim = -1)
            all_predicted = torch.cat((all_predicted, predicted_labels))
            all_labels = torch.cat((all_labels, labels))

            # for i in range(len(predicted_labels)):
            #     print(labels[i].cpu().numpy(), predicted_labels[i].cpu().numpy(), probabilities[i].cpu().numpy())
            print(labels[0].cpu().numpy(), predicted_labels[0].cpu().numpy(), probabilities[0].cpu().numpy(), paths[0])

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")

    tp = conf_matrix[0,0]
    tn = conf_matrix[1,1]
    fp = conf_matrix[1,0]
    fn = conf_matrix[0,1]
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"{len(misclassified_paths) = }, {len(unconfident_paths) = }")

    with open('shadows/pickle/FFT_Dalle_Indoor_Misclassified', 'wb') as f:
        pickle.dump(misclassified_paths, f)
    with open('shadows/pickle/FFT_Dalle_Indoor_Unconfident.pkl', 'wb') as f:
        pickle.dump(unconfident_paths, f)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            make_dataset_shadows.concat_fft(),    
        ])

    misclassified_test_dataset = make_dataset_shadows.DatasetWithFilepaths(misclassified_paths, transform=transform)
    unconfident_test_dataset = make_dataset_shadows.DatasetWithFilepaths(unconfident_paths, transform=transform)
    unconfident_misclassified_test_dataset = ConcatDataset([unconfident_test_dataset, misclassified_test_dataset])

    misclassified_test_loader = DataLoader(dataset=misclassified_test_dataset, batch_size=64, shuffle=False, num_workers=6)
    unconfident_misclassified_test_loader = DataLoader(dataset=unconfident_misclassified_test_dataset, batch_size=64, shuffle=False, num_workers=6)

    full_test(model, misclassified_test_loader, save_to_file="shadows/roc/FFT_Dalle_Indoor_Misclassified", title='ROC for Misclassified Dalle(Indoor) Set')
    full_test(model, unconfident_misclassified_test_loader, save_to_file="shadows/roc/FFT_Dalle_Indoor_Unconfident", title='ROC for Unconfident/Misclassified Dalle(Indoor) Set')
    

def full_test(model, test_dataloader, save_to_file = None, title = "title"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_generated_probs = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for paths, images, labels in tqdm(test_dataloader, desc="testing"):
            images = images.float().to(device)
            labels = labels.to(device)
        
            predictions = model(images)
            
            probabilities = torch.nn.functional.softmax(predictions.data, 1)
            # probabilities = Softmax(dim = 1)(predictions)
            generated_probabilities = probabilities[:, 1]
            predicted_labels = torch.argmax(predictions, dim = -1)
            
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            
            all_labels = torch.cat((all_labels, labels))
            all_generated_probs = torch.cat((all_generated_probs, generated_probabilities))
            all_predicted = torch.cat((all_predicted, predicted_labels))
    
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

    return

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model =  ResNet18_2().to(device)
    save_path = os.path.join('../models','Shadows'+'_'+'indoor')
    model.load_state_dict(torch.load(save_path))
    test_loader = make_dataset_shadows.import_test_data()

    # path, image, label = next(iter(test_loader))
    # print(path, image.shape, label)
    # return

    test_path(model, test_loader)

if __name__ == "__main__":
    main()
