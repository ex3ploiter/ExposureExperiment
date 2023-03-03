import argsparser
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from utills import auc_softmax, auc_softmax_adversarial, save_model_checkpoint, load_model_checkpoint, lr_schedule
from tqdm import tqdm
from torchattacks import FGSM, PGD
from models import Net
from constants import PGD_CONSTANT

def train_one_epoch(epoch, max_epochs, model, optimizer, criterion, trainloader, train_attack, lr=0.1):    
    preds = []
    true_labels = []
    running_loss = 0
    accuracy = 0

    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}/{max_epochs}")

            updated_lr = lr_schedule(learning_rate=lr, t=epoch + (i + 1) / len(list(tepoch)), max_epochs=max_epochs) 
            optimizer.param_groups[0].update(lr=updated_lr)
            
            data, target = data.to(device), target.to(device)
            target = target.type(torch.LongTensor).cuda()
            
            # Adversarial attack on every batch
            data = train_attack(data, target)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            output = model(data)

            # Compute the loss and its gradients
            loss = criterion(output, target)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            true_labels += target.detach().cpu().numpy().tolist()

            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()
            correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
            accuracy = correct / len(preds)

            running_loss += loss.item() * data.size(0)

            tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

    # sample_batch = [first_batch_imgs]
    # attack_names = []

    # for attack_name, attack_module in test_attacks.items():
    #     attack_names.append(attack_name)
    #     sample_batch.append(attack_module(first_batch_imgs, first_batch_labels))

    # writer.add_images(tag='DICk', img_tensor=torch.cat(sample_batch))
    # # writer.add_scalars
    # results["Train Accuracy"].append(100. * accuracy)
    # results["Loss"].append(running_loss / len(preds))
    return accuracy, running_loss / len(preds)


args = argsparser.parse_args()
print(args)

try:
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')


print(device)

if args.model == 'preactresnet18':
    
    model = Net().to(device)

test_attacks = {}

# !python .\train_and_evaluate.py --test_attacks FGSM-8/255 PGD0.03-10 PGD-8/255-100
for test_attack in args.test_attacks:
    try:
        attack_type = test_attack.split('-')[0]
        if attack_type == 'FGSM':
            eps = eval(test_attack.split('-')[1])
            current_attack = FGSM(model, eps=eps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
        elif attack_type == 'PGD':
            eps = eval(test_attack.split('-')[1])
            steps = eval(test_attack.split('-')[2])
            alpha = (PGD_CONSTANT * eps) / steps
            current_attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
    except:
        raise ValueError('Invalid Attack Params!')

# # if desired_attack == 'PGD':
#     train_attack = PGD(model, eps=attack_eps,alpha= pgd_constant * attack_eps / train_attack_steps,steps=attack_steps)
#     test_attack = PGD(model, eps=attack_eps,alpha=pgd_constant * attack_eps / test_attack_steps, steps=attack_steps)
# else:
#     train_attack = FGSM(model, eps=attack_eps)
#     test_attack = FGSM(model, eps=attack_eps)
# train_attack.set_mode_targeted_least_likely()
# test_attack.set_mode_targeted_least_likely()

# train(model, train_loader, test_loader, train_attack, test_attack)
