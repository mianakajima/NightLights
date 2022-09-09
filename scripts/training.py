from scripts.models import *
import torch
import torchvision
import torchvision.transforms as transforms

# get mean and std of image
transform = transforms.Compose(
            [transforms.Resize(size=(430, 430)),
             transforms.Grayscale(1),
             transforms.ToTensor()
             ]
        )
trainset = torchvision.datasets.ImageFolder(root = '../data/training', transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset))
loader_iter = iter(trainloader)
images, _ = loader_iter.next()
image_mean = images.mean()
image_std = images.std()


# prepare data
transform = transforms.Compose(
            [transforms.Resize(size=(430, 430)),
             transforms.Grayscale(1),
             transforms.ToTensor(),
             transforms.Normalize(image_mean, image_std)
             ]
        )

# rotated
transform_rotation = transforms.Compose(
    [transforms.Resize(size = (430, 430)),
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize(image_mean, image_std),
     transforms.RandomRotation(180)
     ]
)

# horizontally flipped
transform_fliph = transforms.Compose(
    [transforms.Resize(size = (430, 430)),
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize(image_mean, image_std),
     transforms.RandomHorizontalFlip(1)
     ]
)

# vertically flipped
transform_flipv = transforms.Compose(
    [transforms.Resize(size = (430, 430)),
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize(image_mean, image_std),
     transforms.RandomVerticalFlip(1)
     ]
)

trainset = torchvision.datasets.ImageFolder(root = '../data/training', transform = transform)

trainset_rotated = torchvision.datasets.ImageFolder(root = '../data/training', transform = transform_rotation)

trainset_fliph = torchvision.datasets.ImageFolder(root = '../data/training', transform = transform_fliph)

trainset_flipv = torchvision.datasets.ImageFolder(root = '../data/training', transform = transform_flipv)

all_training = torch.utils.data.ConcatDataset([trainset, trainset_rotated, trainset_fliph, trainset_flipv])

devset = torchvision.datasets.ImageFolder(root = '../data/dev', transform = transform)

trainloader = torch.utils.data.DataLoader(all_training, batch_size=50,
                                                  shuffle=True, num_workers=2)
devloader = torch.utils.data.DataLoader(devset, batch_size=50, shuffle=False, num_workers=2)

# model
net = CNN()
initial_parameters = net.parameters()
encoder = LitModel(net)

# train model
trainer = pl.Trainer(max_epochs=20, log_every_n_steps=20, accelerator='gpu', devices=1, callbacks=[InputMonitor()])
trainer.fit(model=encoder, train_dataloaders=trainloader, val_dataloaders=devloader)
