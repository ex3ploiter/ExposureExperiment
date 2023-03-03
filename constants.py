# Change this path based on your system paths
CIFAR10_PATH = '~/CIFAR10/'
CIFAR100_PATH = '~/CIFAR100/'
MNIST_PATH = '~/MNIST/'
FMNIST_PATH = '~/FMNIST/'
SVHN_PATH = '~/SVHN/'
MVTEC_PATH = '~/MVTEC/'

PGD_CONSTANT = 2.5

mvtec_labels = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                'wood', 'zipper']


cifar10_super_classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


dataset_labels = {
    'mnist': [str(i) for i in range(10)],
    'svhn': [str(i) for i in range(10)],
    'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'fashion': ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot'],
    'cifar100': ['-'.join(super_class) for super_class in cifar10_super_classes],
    'mvtec': mvtec_labels
}
