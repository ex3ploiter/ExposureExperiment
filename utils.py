from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from argparse import ArgumentParser

def auc_softmax_adversarial(model, test_loader, test_attack, epoch:int, device):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

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

    if is_train:
        model.train()
    else:
        model.eval()

    return auc, accuracy

def auc_softmax(model, test_loader, epoch:int, device):

    is_train = model.training
    model.eval()

    soft = torch.nn.Softmax(dim=1)
    anomaly_scores = []
    preds = []
    test_labels = []

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

  return torch.cat((normal_batch, abnormal_batch),dim=0).cuda(), torch.tensor([0] * n + [1] * n).cuda()


def get_attack_name(attack):
    attack_type = str(attack.__class__)[1:-2].split('.')[-1]
    if attack_type.lower() == 'vanila':
        return attack_type

    attack_str = f'{attack_type} EPS={attack.eps:0.3f}'
    if attack_type.lower() == 'fgsm':
        return attack_str

    attack_str += f' ALPHA={attack.alpha:0.3f} STEPS={attack.steps}'
    return attack_str

def visualize(img_batch, labels, attack, nrow=10):

    ncols = img_batch.shape[0] // nrow


    fig = plt.figure(constrained_layout=True, figsize=(20, (ncols/nrow) * 15 + 2))

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


def parse_args():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Outlier Exposure Experiments Automation')

    parser.add_argument('--source_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'mvtec', 'svhn'], type=str)

    parser.add_argument('--source_class', help='Index of Normal Class',
                        default=None, type=int)

    parser.add_argument('--output_path', help='Path to which plots, results, etc will be recorded',
                        default='./results/', type=str)

    parser.add_argument('--tensorboard_path', help='Path to which plots, results, etc will be recorded on tensorboard',
                        default='./tensorboard/', type=str)

    parser.add_argument('--exposure_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'mvtec', 'svhn', 'adaptive'], type=str)

    parser.add_argument("--checkpoints_path", help='Path to save the checkpoint of trained model', default='./Model-Checkpoints/', type=str)

    parser.add_argument("--max_epochs", help='Maximum number of epochs to Continue training', default=30, type=int)

    parser.add_argument("--batch_size", help='batch_size', default=128, type=int)

    parser.add_argument('--attack_eps', type=str, default='8/255',  help='Attack eps used for both training and testing',)

    parser.add_argument("--pgd_constant", help='PGD Constant', default=2.5, type=float)

    # parser.add_argument('--test_attacks', help='Desired Attacks for adversarial test', nargs='+', action='extend') #Change3

    parser.add_argument('--train_attack_step', help='Desired attack step for adversarial training', default=10, type=int)

    parser.add_argument("--clean", action="store_true", help="if true normal training else adversarial-training")

    parser.add_argument("--force_restart", action="store_true", help="if true doesn't use already available checkpoints")

    parser.add_argument('--test_step', help='If given x, every x step a test would be performed', default=1, type=int)

    parser.add_argument('--save_step', help='If given x, every x step saves a model checkpoint', default=1, type=int)

    parser.add_argument('--cuda_device', help='The number of CUDA device', default=0, type=int)

    parser.add_argument('--loss_threshold', help='The loss threshold which stops training', default=0.001, type=float)

    parser.add_argument('--model', help='Model architecture',
                        choices=['resnet18', 'preactresnet18', 'pretrained_resnet18', \
                                 'resnet34', 'preactresnet34', 'pretrained_resnet34', \
                                 'resnet50', 'preactresnet50', 'pretrained_resnet50', \
                                 'resnet101', 'preactresnet101', 'pretrained_resnet101', \
                                 'resnet152', 'preactresnet152', 'pretrained_resnet152', \
                                 'vit_b_16'], default='preactresnet18', type=str)


    return parser.parse_args()