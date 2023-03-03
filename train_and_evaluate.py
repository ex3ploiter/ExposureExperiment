from distutils.command.config import config
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
from sklearn.metrics import roc_auc_score, accuracy_score



def run(model, checkpoint_path, train_attack, test_attacks, trainloader, testloader, writer, test_step, save_step, max_epochs, loss_threshold=1e-3):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    init_epoch = 0

    checkpoint = load_model_checkpoint(model=model, optimizer=optimizer, path=checkpoint_path)

    if checkpoint is not None:
        model, optimizer, init_epoch, loss = checkpoint

    for epoch in range(init_epoch, max_epochs+1):

        torch.cuda.empty_cache()

        if epoch % test_step == 0 :
                
                test_auc = {}
                test_accuracy = {}

                clean_auc, clean_accuracy  = auc_softmax(model=model, epoch=epoch, test_loader=testloader, device=device)
                test_auc['Clean'], test_accuracy['Clean'] = clean_auc, clean_accuracy

                for attack_name, attack in test_attacks.items():
                    adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=epoch, test_loader=testloader, test_attack=attack, device=device)
                    test_auc[attack_name], test_accuracy[attack_name] = adv_auc, adv_accuracy

                writer.add_scalars('AUC-Test', test_auc, epoch)
                writer.add_scalars('Accuracy-Test', test_accuracy, epoch)
                writer.close()

        torch.cuda.empty_cache()

        train_auc, train_accuracy, train_loss = train_one_epoch(epoch=epoch,\
                                                                max_epochs=max_epochs, \
                                                                model=model,\
                                                                optimizer=optimizer,
                                                                criterion=criterion,\
                                                                trainloader=trainloader,\
                                                                train_attack=train_attack,\
                                                                lr=0.1)
        

        writer.add_scalar('AUC-Train', train_auc, epoch)
        writer.add_scalar('Accuracy-Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy-Loss', train_loss, epoch)
        writer.close()

        if train_loss < loss_threshold:
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint, optimizer=optimizer)
            break
        
        if epoch > 0 and epoch % save_step == 0:
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint, optimizer=optimizer)


def train_one_epoch(epoch, max_epochs, model, optimizer, criterion, trainloader, train_attack, lr=0.1): 

    soft = torch.nn.Softmax(dim=1)

    preds = []
    anomaly_scores = []
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

            probs = soft(output).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

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

    return  roc_auc_score(true_labels, anomaly_scores) , \
            accuracy_score(true_labels, preds, normalize=True), \
            running_loss / len(preds)


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
