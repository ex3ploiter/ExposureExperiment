# ExposureExperiment

### Datasets

|        | **_Cifar10_** | **_Cifar100_** | **_Mnist_** | **_F-Mnist_** | **_MVTEC_** | **_Medical_** | **_SVHN_** |
| ------ | :-----------: | :------------: | :---------: | :-----------: | :---------: | :-----------: | :--------: |
| _eps_  |     8/255     |     8/255      |    8/255    |     8/255     |    2/255    |     2/255     |   8/255    |
| _size_ |     32x32     |     32x32      |    32x32    |     32x32     |   224x224   |    224x224    |   32x32    |

### Example Command to Run

```sh
python train_and_evaluate.py --source_dataset=cifar10 --source_class=0 --source_dataset_path=/CIFAR10/ --exposure_dataset=mnist --exposure_dataset_path=/MNIST/ --output_path=./Results/
```

### Outputs / Run

- Clean AUROC after normal training
- Clean AUROC after adversarial training
- Robust AUROC after adversarial training
- FID Score


### Todo

- fix datasets part and make it clean
- make the pipeline use arguments given by parser
- add save checkpoint option
- add saving results part (both plots for each class and all values of a source dataset in a .csv file)
- choose model acording to the arguments have parsed
- use arguments like source dataset path and others that have been parsed in get_loaders function
