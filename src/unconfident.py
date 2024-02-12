import os
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader
from architectures import ResNet18_2, ResNet50_2
from data import make_dataset_shadows
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import make_dataset_shadows

def test_path(model, test_dataloader, supplement_dataloader):
# def test_path(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # misclassified_paths = []
    # unconfident_paths = []
    unconfident_misclassified_real_paths = []

    # all_predicted = torch.tensor([]).to(device)
    # all_labels = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for paths, images, labels  in tqdm(supplement_dataloader, desc="testing"):
        # for paths, images, labels  in tqdm(test_dataloader, desc="testing"):
        
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            real_probabilities = torch.nn.Softmax(dim = 1)(outputs.data)[:,1]

            # margin = 0.405

            indoor_margin = 0.405
            outdoor_margin = 0.435

            margin = outdoor_margin

            # unconfident_indices_real = (real_probabilities > 0.5) & (real_probabilities < 0.5 + margin) & (labels == 1)
            # unconfident_indices_gen = (real_probabilities < 0.5) & (real_probabilities > 0.5 - margin) & (labels == 0)
            # unconfident_paths += [paths[idx] for idx, val in enumerate(unconfident_indices_real.cpu()) if val]
            # unconfident_paths += [paths[idx] for idx, val in enumerate(unconfident_indices_gen.cpu()) if val]

            # misclassified_indices = ((real_probabilities > 0.5) & (labels == 0)) | ((real_probabilities < 0.5) & (labels == 1))
            # misclassified_paths += [paths[idx] for idx, val in enumerate(misclassified_indices.cpu()) if val]

            unconfident_indices_real = (real_probabilities > 0.5) & (real_probabilities < 0.5 + margin) & (labels == 1)
            unconfident_misclassified_real_paths += [paths[idx] for idx, val in enumerate(unconfident_indices_real.cpu()) if val]

            misclassified_real_indices = (real_probabilities > 0.5) & (labels == 0)
            unconfident_misclassified_real_paths += [paths[idx] for idx, val in enumerate(misclassified_real_indices.cpu()) if val]

            # predicted_labels = torch.argmax(outputs, dim = 1)
            # all_predicted = torch.cat((all_predicted, predicted_labels))
            # all_labels = torch.cat((all_labels, labels))


    # conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    # print(conf_matrix)
    # print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")

    # tp = conf_matrix[0,0]
    # tn = conf_matrix[1,1]
    # fp = conf_matrix[1,0]
    # fn = conf_matrix[0,1]
    # print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    # print(f"{len(misclassified_paths) = }, {len(unconfident_paths) = }")

    # with open('shadows/pickle/FFT_Indoor_Misclassified', 'wb') as f:
    #     pickle.dump(misclassified_paths, f)
    # with open('shadows/pickle/FFT_Indoor_Unconfident.pkl', 'wb') as f:
    #     pickle.dump(unconfident_paths, f)
    # with open('shadows/pickle/FFT_Indoor_Real_Supplement.pkl', 'wb') as f:
    #     pickle.dump(unconfident_misclassified_real_paths, f)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
            make_dataset_shadows.concat_fft(),    
        ])

    # misclassified_test_dataset = make_dataset_shadows.DatasetWithFilepaths(misclassified_paths, transform=transform)
    # unconfident_test_dataset = make_dataset_shadows.DatasetWithFilepaths(unconfident_paths, transform=transform)
    # unconfident_misclassified_test_dataset = ConcatDataset([unconfident_test_dataset, misclassified_test_dataset])

    # misclassified_test_loader = DataLoader(dataset=misclassified_test_dataset, batch_size=64, shuffle=False, num_workers=6)
    # unconfident_misclassified_test_loader = DataLoader(dataset=unconfident_misclassified_test_dataset, batch_size=64, shuffle=False, num_workers=6)

    # full_test(model, misclassified_test_loader, save_to_file="shadows/roc/FFT_Dalle_Indoor_Misclassified", title='ROC for Misclassified Dalle(Indoor) Set')
    # full_test(model, [unconfident_misclassified_test_loader], save_to_file="shadows/roc/FFT_Indoor", title='ROC for Unconfident/Misclassified Indoor Set')

    unconfident_misclassified_real_supplement_dataset = make_dataset_shadows.DatasetWithFilepaths(unconfident_misclassified_real_paths, transform=transform)
    unconfident_misclassified_real_supplement_loader = DataLoader(dataset=unconfident_misclassified_real_supplement_dataset, batch_size=64, shuffle=False, num_workers=6)

    dataloaders = [test_dataloader, unconfident_misclassified_real_supplement_loader]
    full_test(model, dataloaders, save_to_file="shadows/roc/FFT_Kadinsky_Outdoor", title='ROC for Kadinsky(Outdoor) Test Set')
    

def full_test(model, dataloader, save_to_file = None, title = "title"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_generated_probs = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for paths, images, labels in tqdm(dataloader, desc="testing"):
            images = images.float().to(device)
            labels = labels.to(device)
        
            predictions = model(images)
            
            probabilities = torch.nn.Softmax(dim = 1)(predictions.data)
            generated_probabilities = probabilities[:, 0]
            predicted_labels = torch.argmax(predictions, dim = 1)
            
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            
            all_labels = torch.cat((all_labels, labels))
            all_generated_probs = torch.cat((all_generated_probs, generated_probabilities))
            all_predicted = torch.cat((all_predicted, predicted_labels))

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")

    tp = conf_matrix[0,0]
    tn = conf_matrix[1,1]
    fp = conf_matrix[1,0]
    fn = conf_matrix[0,1]
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    fpr, tpr, thresholds = roc_curve(all_labels.cpu(), all_generated_probs.cpu(), pos_label = 0)
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
    model =  ResNet50_2().to(device)
    save_path = '../models/Shadows/Shadows_deepfloyd_indoor_large'
    # save_path = '../models/Shadows/Shadows_kandinsky_indoor_large'
    model.load_state_dict(torch.load(save_path))
    # test_loader = make_dataset_shadows.import_test_data()
    # train_loader, val_loader, test_loader = make_dataset_shadows.import_deepfloyd_indoor_large_data()
    train_loader, val_loader, test_loader = make_dataset_shadows.import_kandinsky_indoor_large_data()
    full_test(model, test_loader, title= 'Deepfloyd Indoor Large eval on Deepfloyd Indoor Large', save_to_file="shadows/roc/FFT_Deepfloyd_Indoor2")

if __name__ == "__main__":
    main()
