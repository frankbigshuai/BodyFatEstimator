import dataclasses
import typing
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter

@dataclasses.dataclass
class Runner:
    model: torch.nn.Module
    criterion: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer
    device: torch.device
    summarywriter: SummaryWriter
    scheduler: torch.optim.lr_scheduler

    def __post_init__(self):
      self.model = self.model.to(device)

    def predict(self, data):
      for X in data:
        X = X.to(self.device)
        output = self.model(X)
        X.cpu()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(output))
        plt.imshow(np.transpose(X.numpy(), (1,2,0)))
        yield output

    def train(self, dataloader, epoch):
      #training loop
      self.model.train()
      train_step = 0
      total_loss = 0
      total_accuracy = 0
      for X,y in dataloader:
        try:
          train_step += 1
          X,y = X.to(self.device), y.to(self.device)
          output = self.model(X)
          #print(y[0],output[0])
          total_accuracy += accuracy(output, y)

          loss = self.criterion(output, y)
          loss.backward()

          self.optimizer.step()
          self.optimizer.zero_grad()

          #run scheduler every batch
          self.scheduler.step()

          total_loss += loss
          yield train_step, loss
        except:
          print('error')

      total_loss = total_loss/train_step
      total_accuracy = total_accuracy/train_step
      self.summarywriter.add_scalar("Loss/training", total_loss, epoch)
      self.summarywriter.add_scalar("Accuracy/training", total_accuracy, epoch)

    def evaluate(self, dataloader, epoch):
      self.model.eval()
      eval_step = 0
      total_loss = 0
      total_accuracy = 0
      total = 0
      with torch.no_grad():
        for X,y in dataloader:
          eval_step += 1
          X,y = X.to(self.device), y.to(self.device)
          output = self.model(X)
          total_accuracy += accuracy(output, y)
          total += y.size(0)
          loss = self.criterion(output, y)
          total_loss += loss
          yield eval_step, loss
      total_loss = total_loss/eval_step
      print(total_loss)
      total_accuracy = total_accuracy/total
      print(total_accuracy)
      self.summarywriter.add_scalar("Loss/evaluate", total_loss, epoch)
      self.summarywriter.add_scalar("Accuracy/evaluate", total_accuracy, epoch)

    def fit(self, epochs, train_loader,  training,val_loader=None, test_loader=None):
      for epoch in range(epochs):
        print(f"EPOCH {epoch + 1}")
        ##train
        for step, loss in self.train(train_loader, epoch):
          print(f"Training - Step: {step} Loss: {loss.item()}")

        #evaluate
        if training: 
          for step, loss in self.evaluate(val_loader, epoch):
            print(f"Validation - Step: {step}| Loss: {loss.item()}")

def accuracy(y_pred, y):
  a  = y - 2 < y_pred 
  b = y_pred < y + 2
  return torch.sum(torch.logical_and(a,b))