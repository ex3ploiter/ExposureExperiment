from ast import arg
from asyncio.log import logger
from asyncore import write
from distutils.command.config import config
import argsparser
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from utills import auc_softmax, auc_softmax_adversarial, save_model_checkpoint, load_model_checkpoint, lr_schedule, get_visualization_batch, visualize, get_attack_name
from tqdm import tqdm
from torchattacks import FGSM, PGD, VANILA
from models import Net
from constants import PGD_CONSTANT, dataset_labels
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import get_dataloader
import os
from logger import Logger

def run(model, checkpoint_path, train_attack, test_attacks, trainloader, testloader, writer, logger:Logger, test_step:int, save_step:int, max_epochs:int, device, force_restart, loss_threshold=1e-3):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    init_epoch = 0

    checkpoint = load_model_checkpoint(model=model, optimizer=optimizer, path=checkpoint_path)

    if not force_restart and checkpoint is not None:
        model, optimizer, init_epoch, loss = checkpoint

    vis_batch_train = get_visualization_batch(dataloader=trainloader, n=50)
    vis_batch_test = get_visualization_batch(dataloader=testloader, n=50)

    writer.add_graph(model, vis_batch_train[0])
    writer.flush()

    logger.add_log(f'Starting Run from epoch {init_epoch}')

    for epoch in range(init_epoch, max_epochs+1):

        torch.cuda.empty_cache()

        logs = {}

        if epoch % test_step == 0 :
                
                test_auc = {}
                test_accuracy = {}

                logger.add_log(f'AUC & Accuracy Vanila - Started...')
                clean_auc, clean_accuracy  = auc_softmax(model=model, epoch=epoch, test_loader=testloader, device=device)
                test_auc['Clean'], test_accuracy['Clean'] = clean_auc, clean_accuracy
                logger.add_log(f'AUC Vanila - score on epoch {epoch} is: {clean_auc * 100}')
                logger.add_log(f'Accuracy Vanila -  score on epoch {epoch} is: {clean_accuracy * 100}')
                logs[f'AUC-Clean'], logs[f'Accuracy-Clean'] = clean_auc, clean_accuracy

                for attack_name, attack in test_attacks.items():
                    logger.add_log(f'AUC & Accuracy Adversarial - {get_attack_name(attack)} - Started...')
                    adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=epoch, test_loader=testloader, test_attack=attack, device=device)
                    test_auc[attack_name], test_accuracy[attack_name] = adv_auc, adv_accuracy
                    logger.add_log(f'AUC Adversairal {get_attack_name(attack)} - score on epoch {epoch} is: {adv_auc * 100}')
                    logger.add_log(f'Accuracy Adversairal {get_attack_name(attack)} -  score on epoch {epoch} is: {adv_accuracy * 100}')
                    logs[f'AUC-{get_attack_name(attack)}'], logs[f'Accuracy-{get_attack_name(attack)}'] = adv_auc, adv_accuracy


                writer.add_scalars('AUC-Test', test_auc, epoch)
                writer.add_scalars('Accuracy-Test', test_accuracy, epoch)
                writer.flush()

                for attack_name, attack in test_attacks.items():
                    fig_train, fig_test =  visualize(vis_batch_train[0], vis_batch_train[1], attack), visualize(vis_batch_test[0], vis_batch_test[1], attack)
                    writer.add_figure(f'Sample Peturbations Train {get_attack_name(attack)}', fig_train, epoch)
                    writer.add_figure(f'Sample Peturbations Test {get_attack_name(attack)}', fig_test, epoch)
                    logger.add_figure(fig=fig_train, epoch=epoch, name='train')
                    logger.add_figure(fig=fig_train, epoch=epoch, name='test')
                    writer.flush()

        torch.cuda.empty_cache()

        logger.add_log(f'Starting Training on epoch {init_epoch}')

        train_auc, train_accuracy, train_loss = train_one_epoch(epoch=epoch,\
                                                                max_epochs=max_epochs, \
                                                                model=model,\
                                                                optimizer=optimizer,
                                                                criterion=criterion,\
                                                                trainloader=trainloader,\
                                                                train_attack=train_attack,\
                                                                lr=0.1,\
                                                                device=device)
        
        writer.add_scalar('AUC-Train', train_auc, epoch)
        writer.add_scalar('Accuracy-Train', train_accuracy, epoch)
        writer.add_scalar('Train-Loss', train_loss, epoch)
        writer.flush()

        logs['AUC-Train'],  logs['Accuracy-Train'], logs['Train-Loss'] = train_auc, train_accuracy, train_loss
        logger.add_csv(dict_to_append=logs)

        if train_loss < loss_threshold:
            logger.add_log(f'Early Stopping! the train loss is lower than {loss_threshold}')
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint_path, optimizer=optimizer)
            break
        
        if epoch > 0 and epoch % save_step == 0:
            logger.add_log(f'Saved Model on epoch {epoch} at {checkpoint_path}')
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint_path, optimizer=optimizer)

    logger.add_log(f'Run successfully finished!')
    writer.close()

def train_one_epoch(epoch, max_epochs, model, optimizer, criterion, trainloader, train_attack, lr, device): 

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
            updated_lr = lr_schedule(learning_rate=lr, t=epoch + (i + 1) / len(tepoch), max_epochs=max_epochs) 
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

    return  roc_auc_score(true_labels, anomaly_scores) , \
            accuracy_score(true_labels, preds, normalize=True), \
            running_loss / len(preds)


##################
#  Parsing Args  #
##################

args = argsparser.parse_args()


#######################
#  init custom logger #
#######################

logger_dir = os.path.join(args.output_path, "Clean-Train" if args.clean else "Adversarial-Train", f'normal-{args.source_dataset}', f'normal-class-{args.source_class:02d}-{dataset_labels[args.source_dataset][args.source_class]}', f'exposure-{args.exposure_dataset}')
experiment_name = f'{args.model}-{args.source_dataset}-{args.source_class:02d}--{args.exposure_dataset}'
logger = Logger(save_dir=logger_dir, exp_name=experiment_name, hparams=args)

logger.add_log(args)


################
#  Set Device  #
################

device = None

try:
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')


logger.add_log(device)

####################
#  Model Selection #
####################

model = None

try:
    model = Net(args.model).to(device)
except Exception as err:
    raise err

logger.add_log(args.model)


#####################
#  Attacks Eps Init #
#####################

attack_eps = None

try:
    attack_eps = eval(args.attack_eps)
except:
    raise ValueError('Wrong Epsilon Value!')

######################
#  Test Attacks Init #
######################

# !python .\train_and_evaluate.py --test_attacks FGSM PGD-10 PGD-100

test_attacks = {}


for test_attack in args.test_attacks:
    try:
        attack_type = test_attack.split('-')[0] if test_attack != 'FGSM' else 'FGSM'
        if attack_type == 'FGSM':
            current_attack = FGSM(model, eps=attack_eps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
        elif attack_type == 'PGD':
            steps = eval(test_attack.split('-')[1])
            alpha = (PGD_CONSTANT * attack_eps) / steps
            current_attack = PGD(model, eps=attack_eps, alpha=alpha, steps=steps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
    except:
        raise ValueError('Invalid Attack Params!')


######################
#  Train Attack Init #
######################

train_steps = args.train_attack_step
train_alpha = (PGD_CONSTANT * attack_eps) / train_steps
train_attack = PGD(model, eps=attack_eps, alpha=train_alpha, steps=train_steps) if not args.clean else VANILA(model)


################
#  Dataloaders #
################

trainloader, testloader = get_dataloader(normal_dataset=args.source_dataset, normal_class_indx=args.source_class, exposure_dataset=args.exposure_dataset, batch_size=args.batch_size)

#########################
#  init checkpoint path #
#########################

checkpoint_dir = os.path.join(args.checkpoints_path, "Clean-Train" if args.clean else "Adversarial-Train", f'normal-{args.source_dataset}', f'normal-class-{args.source_class:02d}-{dataset_labels[args.source_dataset][args.source_class]}', f'exposure-{args.exposure_dataset}')
checkpoint_name = f'{args.model}-{args.source_dataset}-{args.source_class:02d}--{args.exposure_dataset}.pt'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)


############################
#  init tensorboard writer #
############################

writer_dir = os.path.join(args.tensorboard_path, "Clean-Train" if args.clean else "Adversarial-Train", f'normal-{args.source_dataset}', f'normal-class-{args.source_class:02d}-{dataset_labels[args.source_dataset][args.source_class]}', f'exposure-{args.exposure_dataset}')
writer = SummaryWriter(writer_dir)


##################################
#               RUN              #
##################################


run(model=model,\
    checkpoint_path=checkpoint_path,\
    train_attack=train_attack,\
    test_attacks=test_attacks,\
    trainloader=trainloader,\
    testloader=testloader,\
    writer=writer,\
    logger=logger,\
    test_step=args.test_step,\
    save_step=args.save_step,\
    max_epochs=args.max_epochs,\
    device=device,\
    force_restart=args.force_restart,\
    loss_threshold=args.loss_threshold\
    )