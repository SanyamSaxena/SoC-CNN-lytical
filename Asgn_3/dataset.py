import torchvision.transforms as transforms
from torch.utils.data import Dataset

# define your dataset class
class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.img_labels = y
        self.imgs = X
        T0 = transforms.ToPILImage()
        # T1 = transforms.RandomCrop(32, padding=4)
        T2 = transforms.RandomHorizontalFlip()
        T3 = transforms.ToTensor()
        # using these values for normalising different channels since these are globally available value for CIFAR 10 dataset
        nr = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        # Transforms object for trainset with augmentation
        transform_with_aug = transforms.Compose([T0, T2, T3, nr])
        
        self.transform = transform_with_aug
        
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
          image = self.transform(image)
        return image, label
