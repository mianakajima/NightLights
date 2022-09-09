import unittest
from scripts.data_helpers_night_lights import *
from scripts.models import *
import torch
import torchvision
import torchvision.transforms as transforms

class TestDataHelpers(unittest.TestCase):

    def test_calibration(self):

        c0, c1, c2 = get_calib_coefficients('testtestF152007', 2007)
        self.assertEqual(c0, 1.3606)
        self.assertEqual(c1, 1.2974)
        self.assertEqual(c2, -0.0045)

    def test_model_is_learning(self):

        transform = transforms.Compose(
            [transforms.Resize(size=(430, 430)),
             transforms.Grayscale(1),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)
             ]
        )

        trainset = torchvision.datasets.ImageFolder(root='../data/training', transform=transform)
        devset = torchvision.datasets.ImageFolder(root='../data/dev', transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                                  shuffle=True, num_workers=2)

        devloader = torch.utils.data.DataLoader(devset, batch_size=50, shuffle=False, num_workers=2)


        # model
        net = CNN()
        encoder = LitModel(net)
        initial_parameters = encoder.parameters()
        # train model
        trainer = pl.Trainer(max_epochs=1, log_every_n_steps=20, accelerator="gpu")
        trainer.fit(model=encoder, train_dataloaders=trainloader, val_dataloaders=devloader)

        end_net_parameters = encoder.parameters()

        assert initial_parameters != end_net_parameters

if __name__ == '__main__':
    unittest.main()
