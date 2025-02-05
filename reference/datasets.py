import torch
import os
import pandas as pd
import numpy as np
import torchvision

class BodyFatImgDataset(torch.utils.data.Dataset):
    """Body fat dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train = False):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      super().__init__()
      self.root_dir = root_dir
      file_name = os.path.join(self.root_dir,
                              csv_file)
      self.bodyfats = pd.read_csv(file_name)
      self.transform = transform # store transforms 
      length = len(self.bodyfats) # length of total dataset 
      train_length = np.floor(length*0.8).astype('int') # length of training dataset
      if train:
        self.dataset = self.split(0,train_length)
      else:
        self.dataset = self.split(train_length, length)

    def __len__(self):
      return len(self.dataset["bodyfat"])

    def __getitem__(self, idx):
      '''
      defines get item to return item by indexing dataset class 
      input : index(int)
      return : image(torch.tensor), bodyfat(float)
      '''
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = os.path.join(self.root_dir,
                              self.dataset["image"][idx]) # img path
      image = torchvision.io.read_image(path=img_name,mode = torchvision.io.image.ImageReadMode.RGB ) # get img from path
      image = image.to(torch.float)
      if self.transform:
        image = self.transform(image)
      bodyfat = torch.tensor([self.dataset["bodyfat"][idx]],dtype=torch.float) # retrieve targets

      return image, bodyfat 

    def get_list(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()

      img_name = os.path.join(self.root_dir,
                              self.bodyfats.iloc[idx, 0]+'.jpg') # img path
      bodyfat = self.bodyfats.iloc[idx, 1].astype('float') # retrieve targets
      sample = {'image': img_name, 'bodyfat': bodyfat} 

      return sample

    def split(self, id1, id2):
      dics = []
      d = {}
      for i in range(id1,id2): 
        sample = self.get_list(i) # get samples from id1 to id2
        dics.append(sample) # add them to dict dics
      for k in dics[0].keys():
        d[k] = list(f[k] for f in dics)
      return d