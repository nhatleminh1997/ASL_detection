import torch
import torch.utils.data as data
import os
# import glob
import string
import pathlib
from torchvision.io import read_image
class ASL(data.Dataset):
    """
    Args: 
        root_path:   (str) root of extracted data contains train and test set
        subset: (str) either 'Train' or 'Test' for corresponding data subset
        dim: (int) 1 to get dataset as sequence data, 2 to get data as 1-channel 2D image
    """

    def __init__(self, root_path, subset, dim = 2):
        self.subset = subset 
        self.root_path = os.path.join(root_path, subset)
        self.all_images = list(pathlib.Path(self.root_path).rglob('*.jpg'))
        self.all_images = [str(i) for i in self.all_images]

        # process label 
        self.char_classes = [char for char in string.ascii_uppercase if char != "J" if char != "Z"]
        self.length = len(self.all_images)
        self.dim = dim
        
    def __getitem__(self, index):
        """
        Args: 
            root_path:   (str) root of extracted data contains train and test set
            subset: (str) either 'Train' or 'Test' for corresponding data subset
        """
        img_path = self.all_images[index]
        # print(img_path)
        img = read_image(img_path).float()
        img = img/255. - 0.5
        if self.dim == 1:
            img = torch.flatten(img, start_dim=1)
        char_label = img_path.split('.')[-2][-1]
        num_label = self.char_classes.index(char_label)
        return img, num_label

    def __len__(self):
        return self.length

# all_images = []
root_path = r'D:\ML_final\Data'
subset = 'Train'
# # for root, dirs, files in os.walk(root_path):
# #     for file in files:
# #         if file.endswith(".jpg"):
# #             all_images.append(file)

# all_images = list(pathlib.Path(root_path).rglob('*.jpg'))

# print(all_images)
# print(len(all_images))

# classes = [char for char in string.ascii_uppercase if char != "J" if char != "Z"]
# print(classes)

### Testing

# train_set = ASL(root_path, subset, dim = 1)
# img, num_label = train_set[4000]
# print(img.shape)
# print(num_label)
# print(len(train_set))
