import torch
from torch.utils.data import Dataset, DataLoader






from PIL import Image


import torchvision.transforms as transforms
from os import listdir


class Image_data(Dataset):
    def __init__(self, data_dir, low, high, phase='train'):
        super(Image_data, self).__init__()
        self.low = low
        self.high = high
        self.phase = phase
        self.low_dir = data_dir + '/' + str(high) + '/' + phase + '/from_' + str(low)
        self.high_dir = data_dir + '/' + str(high) + '/' + phase + '/original'
        self.ids = listdir(self.low_dir)

    def __getitem__(self, idx):
        id = self.ids[idx]
        low_file = self.low_dir + '/' + id
        high_file = self.high_dir + '/' + id
        low = Image.open(low_file).convert("RGB")
        high = Image.open(high_file).convert("RGB")
        low_t = transforms.ToTensor()(low)
        high_t = transforms.ToTensor()(high)
        return {
            'SR': low_t.type(torch.FloatTensor),
            'HR': high_t.type(torch.FloatTensor)
        }

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data/cells')
    trainset = Image_data(data_dir=data_dir)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    for i, data in enumerate(trainloader):
        high, low = data
    for i in range(trainset.__len__()):
        high, low = trainset.__getitem__(i)['high'], trainset.__getitem__(i)['low']