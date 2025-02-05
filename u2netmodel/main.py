from main.datasets import BodyFatImgDataset
from dataloaders import DataLoad
from runner import Runner
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

root_dir = "/content/drive/.shortcut-targets-by-id/1IhsQD83MoJF9RQAIwUlCCG3p1fV8cUo8/Kayse_Hussein_Colab/Data/Images"

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.GaussianBlur((5, 9), sigma=(0.1, 2.0)),
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = BodyFatImgDataset(
    csv_file="Labels.csv",
    root_dir=root_dir,
    transform=train_transform,
    train=True,
)

test_dataset = BodyFatImgDataset(
    csv_file="Labels.csv",
    root_dir=root_dir,
    transform=test_transform,
    train=False,
)

train_dataset, validation_dataset = torch.utils.data.random_split(
    train_dataset, [650, 167]
)

Train_Batch_Size = 16  # multiple steps to increase randomness
Val_Batch_Size = 64
Test_Batch_Size = 205


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoad()
    train_loader = loader.train_loader(
        train_dataset, Train_Batch_Size=Train_Batch_Size)
    val_loader = loader.train_loader(
        validation_dataset, Train_Batch_Size=Val_Batch_Size)
    test_loader = loader.train_loader(
        test_dataset, Train_Batch_Size=Test_Batch_Size)
    channels=3
    criterion = torch.nn.MSELoss()

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features= 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=512, out_features=1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1024, out_features=1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=1024, out_features=1)
    )

    param_groups = [
            {'params':model.layer1.parameters(),'lr':.0001},
            {'params':model.layer2.parameters(),'lr':.0001},
            {'params':model.layer3.parameters(),'lr':.0001},
            {'params':model.layer4.parameters(),'lr':.0001},
            {'params':model.fc.parameters(),'lr':.003}
    ]

    optimizer = torch.optim.SGD(param_groups)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], verbose = True)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/100, max_lr=lr, step_size_up = 55 ,verbose=True, cycle_momentum=False) 
    summarywriter = SummaryWriter()
    runner = Runner(model, criterion, optimizer, device, summarywriter, lr_scheduler)
    epochs = 15
    runner.fit(epochs, train_loader,training= True, val_loader=val_loader)
    
