# Description: Contains Utility class for loading and preparing the image dataset for training
# author: Kolade Gideon @Allaye
# github: www.github.com/allaye
# created: 2023-03-06
# last modified: 2023-03-25


from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import returns_files_in_order


#
# x = torch.rand(5, 3)
# print(x)
# print(torch.cuda.is_available())


class DenoiserDataset(Dataset):
    """
    class
    """
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.noisy_images_uris = returns_files_in_order(self.path[0])
        self.clean_images_urls = returns_files_in_order(self.path[1])

    def getimages(self, idx):
        noisy_image, clean_image = self.__getitem__(idx)
        img1 = transforms.ToPILImage()(noisy_image[0])
        img2 = transforms.ToPILImage()(clean_image[0])
        img1.show()
        img2.show()

    def __len__(self):
        """
        override the len method to return the length of the dataset
        :return: returns the length of the dataset
        """
        return len(self.noisy_images_uris)

    def __getitem__(self, idx):
        """
        override the getitem method to return the noisy and clean images
        :param idx: index of the image url
        :return: both the noisy and clean images
        """
        # we can check if the index is out of bounds here, but we don't need to
        noisy_image = Image.open(self.noisy_images_uris[idx])
        clean_image = Image.open(self.clean_images_urls[idx])
        if self.transform is not None:
            noisy_image = self.transform(noisy_image.resize((128, 128)))
            clean_image = self.transform(clean_image.resize((128, 128)))
        return noisy_image, clean_image


# from PIL import Image
# from tqdm import tqdm

#
# if __name__ == "__main__":
#     dataset1 = DenoiserDataset(['../data/noisey_train/*.jpg', '../data/train/*.jpg'], transform=transforms.ToTensor())
#     for i in range(1000):
#         dataset1.getimages(i)
#         print(i)
# #
#     data = DataLoader(dataset1, batch_size=4, shuffle=True)
#     print(type(data), len(data))
#     for idx, (noisy, clean) in enumerate(data):
#         print('noisy', type(noisy), noisy.shape, clean.shape)
#         # image = Image.fromarray(noisy.numpy().astype('uint8'))
#         img1 = transforms.ToPILImage()(noisy[idx])
#         img2 = transforms.ToPILImage()(clean[idx])
#         img1.show()
#         img1.size
#         img2.show()
#         img2.size
# #         # print('clean', len(clean))
# #         # #image = Image.fromarray(clean.numpy().astype('uint8'))
# #         # image.show()
# #
