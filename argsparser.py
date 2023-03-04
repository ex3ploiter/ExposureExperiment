from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Outlier Exposure Experiments Automation')

    parser.add_argument('--source_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'mvtec-ad', 'med'], type=str)

    parser.add_argument('--source_class', help='Index of Normal Class',
                        default=None, type=int)

    parser.add_argument('--output_path', help='Path to which plots, results, etc will be recorded',
                        default='./results/', type=str)
    
    parser.add_argument('--tensorboard_path', help='Path to which plots, results, etc will be recorded on tensorboard',
                        default='./tensorboard/', type=str)
    
    parser.add_argument('--exposure_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'mvtec', 'svhn'], type=str)

    parser.add_argument("--checkpoints_path", help='Path to save the checkpoint of trained model', default='./Model-Checkpoints/', type=str)

    parser.add_argument("--max_epochs", help='Maximum number of epochs to Continue training', default=30, type=int)

    parser.add_argument("--batch_size", help='batch_size', default=128, type=int)
    
    parser.add_argument('--attack_eps', type=str, default='8/255',  help='Attack eps used for both training and testing',)

    parser.add_argument('--test_attacks', help='Desired Attacks for adversarial test', nargs='+', action='extend')

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
                                 'vit'], default='preactresnet18', type=str)
    

    return parser.parse_args()