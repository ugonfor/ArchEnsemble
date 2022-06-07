from PIL import Image
import torchvision.datasets as datasets
import torch

class MIMOCollator:
    def __init__(self):
        pass
    
    def __call__(self, batch):
        print(type(batch))
        print(batch[0][0].shape)
        print(batch[0][1])
        print(len(batch[0]))
        return batch


class MIMOCifar10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mimo=1):
        super().__init__(root, train, transform, target_transform, download)
        self.mimo = mimo
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        imgs = []
        _targets = []

        for i in range(self.mimo):
            idx = self.mimo * index + i
            img, target = self.data[idx], self.targets[idx]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            imgs.append(img)
            _targets.append(target)

        return torch.cat(imgs,dim=0), torch.tensor(_targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)//self.mimo