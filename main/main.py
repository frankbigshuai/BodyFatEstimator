
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import BodyFatDataset  # 导入自定义数据集类
from DataLoader import create_dataloaders  # 导入创建 DataLoader 的函数
from resnet import resnet50  # 导入自定义 ResNet 模型

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 初始化学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 创建保存目录
        self.save_dir = os.path.join('models', datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化最佳指标
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
    def freeze_layers(self):
        """冻结除 fc 层外的所有层"""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
    def unfreeze_layers(self, num_layers):
        """解冻最后 num_layers 层"""
        layers = list(self.model.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self, loader, desc="Validating"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': total_loss/(batch_idx+1),
                    'acc': 100.*correct/total
                })
                
        return total_loss/len(loader), 100.*correct/total
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            
    def train(self):
        logger.info("Starting training...")
        patience = self.config['patience']
        patience_counter = 0
        
        # 首先只训练 fc 层
        self.freeze_layers()
        logger.info("Frozen all layers except fc")
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # 逐步解冻
            if epoch == self.config['unfreeze_epoch']:
                logger.info("Unfreezing last 3 layers")
                self.unfreeze_layers(3)  # 解冻最后 3 层
                
            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch()
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            
            # 验证
            val_loss, val_acc = self.validate(self.val_loader)
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                patience_counter = 0
                logger.info(f'New best model! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
            
            # 早停
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {patience} epochs without improvement')
                break
                
        # 训练结束后在测试集上评估
        logger.info("\nTraining finished! Evaluating on test set...")
        test_loss, test_acc = self.validate(self.test_loader, desc="Testing")
        logger.info(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        return self.best_val_acc, test_acc

def main():
    # 配置参数
    config = {
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'patience': 5,
        'unfreeze_epoch': 5,  # 在第 5 个 epoch 解冻
    }
    
    # 数据加载
    csv_file = '/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/label/cleaned_data_male.json'
    root_dir = '/Volumes/TOSHIBA/bodypicsdataset/zipevery1000/photos_batch_1/male'
    batch_size = 32
    
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=batch_size
    )
    
    # 初始化模型
    model = resnet50(num_classes=9)
    
    # 加载预训练权重
    checkpoint = torch.load(
        "/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/main/resnet50.pth",
        weights_only=True  # 确保安全加载
    )
    
    # 去掉 fc 层的权重
    pretrained_dict = {k: v for k, v in checkpoint.items() if "fc" not in k}
    
    # 加载到模型中
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    # 重新初始化 fc 层
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # 开始训练
    best_val_acc, test_acc = trainer.train()
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()