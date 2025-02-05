import torch
class DataLoad():
    '''
    Define the torch dataloaders for each dataset

    '''
    def train_loader(self,train_dataset, Train_Batch_Size):
        return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Train_Batch_Size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    def val_loader(self, validation_dataset, Val_Batch_Size):
        return torch.utils.data.DataLoader(
        validation_dataset, batch_size=Val_Batch_Size, drop_last=True, pin_memory=torch.cuda.is_available()
    )

    def test_loader(self, test_dataset, test_Batch_Size):
        return torch.utils.data.DataLoader(
        test_dataset, batch_size=test_Batch_Size, drop_last=True, pin_memory=torch.cuda.is_available()
    )
