import albumentations as A
from torch.utils.data import Dataset

# define your dataset class
class ImageDataset(Dataset):
    def __init__(self, X, y):

        self.img_masks = y
        self.imgs = X
        transform_with_aug = A.Compose(
            [   A.Rotate(limit=45, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.pytorch.ToTensor(),            
                ]
        )
             
        # transform_2=A.compose(
        #     [        A.Normalize(
        #              mean=[0.485, 0.456, 0.406],
        #              std=[0.229, 0.224, 0.225]),
        #             ),
        #     ]
        # )   

        self.transform = transform_with_aug
        
    def __len__(self):
        return len(self.img_masks)
        
    def __getitem__(self, idx):
        image = self.imgs[idx]
        mask = self.img_masks[idx]
        if self.transform:
          augmentations = self.transform(image=image, mask=mask)
          image = augmentations["image"]
          mask = augmentations["mask"]
        return image, mask


