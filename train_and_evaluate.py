import argsparser
import torch
from torch import optim
from utills import auc_softmax, auc_softmax_adversarial
from tqdm import tqdm
from torchattacks import FGSM, PGD
from torchvision.utils import save_image
from torchvision.utils import make_grid
from models import FeatureExtractor
from constants import PGD_CONSTANT
# def train(model, trainloader, testloader, train_attack, test_attack, lr=1e-4, weight_decay=5e-5):
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     for epoch in range(1, n_epochs+1):
#         print(f"Epoch {epoch} started")
        
#         total_loss, total_num = 0.0, 0
#         model.train()
#         loss = nn.CrossEntropyLoss()
#         train_bar =  tqdm(trainloader, desc='Train Binary Classifier ...')
#         for (img1, Y ) in train_bar:
#             adv_data = train_attack(img1, Y)
#             optimizer.zero_grad()
#             out_1 = model(adv_data.cuda()) 
#             loss_ = loss(out_1,Y.cuda())  
#             loss_.backward()
#             optimizer.step()
#             total_num += img1.size(0)
#             total_loss += loss_.item() * img1.size(0)
#             total_num += trainloader.batch_size
#             total_loss += loss_.item() * trainloader.batch_size
#             train_bar.set_description('Train Robust Epoch :  {} , Clf_B Robust Loss: {:.4f}'.format(epoch ,  total_loss / total_num))

#         if (epoch + 1) % auc_every == 0:
#             print(f"Starting to Test After {epoch} epochs have finished:")
#             model.eval()
#             auc_softmax(model, testloader, epoch)
#             auc_softmax_adversarial(model, testloader, test_attack, epoch)

args = argsparser.parse_args()
print(args)
try:
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')


print(device)

if args.model == 'preactresnet18':
    
    model = FeatureExtractor().to(device)

test_attacks = []

# !python .\train_and_evaluate.py --test_attacks FGSM-8/255 PGD0.03-10 PGD-8/255-100
for test_attack in args.test_attacks:
    try:
        attack_type = test_attack.split('-')[0]
        if attack_type == 'FGSM':
            eps = eval(test_attack.split('-')[1])
            current_attack = FGSM(model, eps=eps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks.append(current_attack)
        elif attack_type == 'PGD':
            eps = eval(test_attack.split('-')[1])
            steps = eval(test_attack.split('-')[2])
            alpha = (PGD_CONSTANT * eps) / steps
            current_attack = PGD(model, eps=eps,alpha=alpha, steps=steps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks.append(current_attack)
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
