
class MyDataset_Binary(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, labels,transform):
        'Initialization'
        super(MyDataset_Binary, self).__init__()
        self.labels = labels
        self.x = x
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.transform(self.x[index])
        y = self.labels[index]
       
        return x, y

def get_loaders(dataset, exposure_dataset='cifar10' train_batch_size=40, test_batch_size=300):

    trans_to_32 = transforms.Compose([  transforms.Resize((32,32))])

    trans_to_28 = transforms.Compose([  transforms.Resize((28,28))])


    img_transform_32 = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])

    img_transform_simple = transforms.Compose([transforms.Resize((32,32)), transforms.Grayscale(3), transforms.ToTensor()])

    simple_train_ds_MNIST =  MNIST(root = './', train=True, download=True, transform=img_transform_simple)
    cnt_normal = sum(np.asarray(simple_train_ds_MNIST.targets) == normal_class)

    idx = (simple_train_ds_MNIST.targets==normal_class) 
    simple_train_ds_MNIST.targets = simple_train_ds_MNIST.targets[idx]
    simple_train_ds_MNIST.data = simple_train_ds_MNIST.data[idx]

    images_train = []
    for x,_ in simple_train_ds_MNIST:
        images_train.append(x)
    MNIST_images_train_t = torch.stack(images_train)

    if len(exposures) == 2:
        MNIST_images_train_t = np.concatenate([MNIST_images_train_t.numpy(),MNIST_images_train_t.numpy()])
        MNIST_images_train_t = torch.from_numpy(MNIST_images_train_t)

    images_train_t = MNIST_images_train_t
    ########

    if "gluide" in exposures:
        loaded = np.load(fakes_path)
        generated_fakes = trans_to_32(trans_to_28(torch.from_numpy(loaded)))
        scaled = ((generated_fakes + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
        generated_fakes = scaled/255

        gray_scale_transformation_tr = torchvision.transforms.Grayscale(num_output_channels=3)
        generated_fakes = gray_scale_transformation_tr(generated_fakes)[:cnt_normal]

    if "cifar10" in exposures:
        Cifar10_testset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform_32)

        testset_cifar10=[]
        for x,_ in Cifar10_testset:
            testset_cifar10.append(x)
        Cifar10_test_tensor = torch.stack(testset_cifar10[:cnt_normal])

    if "gluide" in exposures and "cifar10" in exposures:
        Fakes = torch.from_numpy(np.concatenate([generated_fakes.numpy() , Cifar10_test_tensor.numpy()]))
    elif "gluide" in exposures:
        Fakes = generated_fakes
    elif "cifar10" in exposures:
        Fakes = Cifar10_test_tensor

    simple_test_ds =  MNIST(root = './', train=False, download=True, transform=img_transform_simple)
    test_labels=[]
    images_test=[]

    for x,y in simple_test_ds:
        images_test.append(x)
        if y==normal_class:
            test_labels.append(0)
        else:
            test_labels.append(1)
    images_test_t = torch.stack(images_test)

    print(images_train_t.shape[0])

    Binary_Trainig_ = torch.cat((images_train_t , Fakes[:images_train_t.shape[0]]))
    normlabel_ = [0] * images_train_t.shape[0]
    abnormal_label_ = [1] * images_train_t.shape[0]
    labels_train = normlabel_ + abnormal_label_
    mir_data_set_Binary = MyDataset_Binary(Binary_Trainig_, labels_train, trans_to_32)
    train_loader = DataLoader(mir_data_set_Binary, train_batch_size=40, shuffle=True)



    mir_data_set_test = MyDataset_Binary(images_test_t, test_labels, trans_to_32)
    test_loader = DataLoader(mir_data_set_test, test_batch_size=300, shuffle=False)
    ######

    show_samples(Fakes[:64])
    show_samples(images_train_t[:64])


    print("This is len of Train normal :",len(train_loader.dataset))
    print("This is len of test set:",len(test_loader.dataset))

    return train_loader, test_loader

