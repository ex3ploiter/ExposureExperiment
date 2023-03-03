from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid

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
    

def get_visualization_batch(dataloader, n):
  
  iterator = iter(dataloader)
  images_batch, labels_batch = next(iterator)
  while True:
    if labels_batch.sum().item() > n and (1 - labels_batch).sum().item() > n:
      break

    new_images_batch, new_labels_batch = next(iterator)
    labels_batch = torch.cat((labels_batch, new_labels_batch), dim=0)
    images_batch = torch.cat((images_batch, new_images_batch), dim=0)

  normal_batch = images_batch[labels_batch==0][:n]
  abnormal_batch = images_batch[labels_batch==1][:n]

  return torch.cat((normal_batch, abnormal_batch),dim=0), torch.tensor([0] * n + [1] * n)


def get_attack_name(attack):
    attack_type = str(attack.__class__)[1:-2].split('.')[-1]
    attack_str = f'{attack_type} EPS={attack.eps:0.3f}'
    if attack_type.lower() == 'fgsm':
        return attack_str
    attack_str += f' ALPHA={attack.alpha:0.3f} STEPS={attack.steps}'
    return attack_str

def visualize(img_batch, labels, attack, nrow=10):

    ncols = img_batch.shape[0] // nrow

    fig = plt.figure(constrained_layout=True, figsize=(20, ncols * 15 + 2))
    
    fig.suptitle(get_attack_name(attack), size=32)

    subfigs = fig.subfigures(nrows=3, ncols=1)

    adv_batch = attack(img_batch, labels)
    noise_batch =  adv_batch - img_batch
    noise_batch = (noise_batch - torch.min(noise_batch))/(torch.max(noise_batch) - torch.min(noise_batch))

    batchs = [img_batch, adv_batch, noise_batch]
    titles = ['Clean', 'Purturbed', 'Normalized Noise']

    for subfig, batch, title in zip(subfigs, batchs, titles):
        subfig.suptitle(title, size=23)

        axs = subfig.subplots(nrows=1, ncols=2)
        
        batch_shape = batch.shape[0]
        normal_images, adversarial_images = batch[:batch_shape//2], batch[batch_shape//2:]

        axs[0].plot()
        img = F.to_pil_image(make_grid(normal_images, nrow=nrow))
        axs[0].imshow(np.array(img))
        axs[0].grid(False)
        axs[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        axs[1].plot()
        img = F.to_pil_image(make_grid(adversarial_images, nrow=nrow))
        axs[1].imshow(np.array(img))
        axs[1].grid(False)
        axs[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    return fig