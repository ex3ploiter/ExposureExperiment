from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch
from tqdm import tqdm

def auc_softmax_adversarial(model, test_loader, test_attack, epoch:int, device):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    print('AUC & Accuracy Adversarial Softmax Started ...')

    with tqdm(test_loader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)

            adv_data = test_attack(data, target)
            output = model(adv_data)
            
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()

            probs = soft(output).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

            test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    accuracy = accuracy_score(test_labels, preds, normalize=True)

    print(f'AUC Adversairal - Softmax - score on epoch {epoch} is: {auc * 100}')
    print(f'Accuracy Adversairal - Softmax - score on epoch {epoch} is: {accuracy * 100}')

    if is_train:
        model.train()
    else:
        model.eval()

    return auc

def auc_softmax(model, test_loader, epoch:int, device):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

    print('AUC & Accuracy Softmax Started ...')
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            torch.cuda.empty_cache()
            for i, (data, target) in enumerate(tepoch):
                data, target = data.to(device), target.to(device)
                output = model(data)

                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                preds += predictions.detach().cpu().numpy().tolist()

                probs = soft(output).squeeze()
                anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

                test_labels += target.detach().cpu().numpy().tolist()

    auc = roc_auc_score(test_labels, anomaly_scores)
    accuracy = accuracy_score(test_labels, preds, normalize=True)

    print(f'AUC - Softmax - score on epoch {epoch} is: {auc * 100}')
    print(f'Accuracy - Softmax - score on epoch {epoch} is: {accuracy * 100}')

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy

def save_model_checkpoint(model, epoch, loss, path, optimizer):
    try:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
        }, path)
    except:
        raise ValueError('Saving model checkpoint failed!')
    

def load_model_checkpoint(model, optimizer, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    except:
        return None


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


def lr_schedule(learning_rate:float, t:float, max_epochs:int):
    if t / max_epochs < 0.5:
        return learning_rate
    elif t / max_epochs < 0.75:
        return learning_rate / 10.
    else:
        return learning_rate / 100.