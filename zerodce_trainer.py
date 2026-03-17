import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import Zero-DCE model
from zero_dce_model import ZeroDCENet

class ZeroDCETrainer:
    """
    Comprehensive Zero-DCE Training Pipeline
    Focus on thorough training and hyperparameter optimization
    """
    
    def __init__(self, model=None, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model if model else ZeroDCENet(iteration=8).to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        self.best_loss = float('inf')
        
        print(f"Zero-DCE Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-6, scheduler_type='cosine'):
        """
        Setup optimizer and scheduler with different options
        """
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, factor=0.5)
        
        print(f"Training setup: lr={learning_rate}, scheduler={scheduler_type}")
    
    def zero_dce_loss(self, enhanced, original, curves):
        """
        Enhanced Zero-DCE loss function with all components
        """
        # 1. Spatial Consistency Loss
        def spatial_consistency_loss(img):
            gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
            grad_x = torch.mean(torch.abs(gray[:, :, 1:] - gray[:, :, :-1]))
            grad_y = torch.mean(torch.abs(gray[:, 1:, :] - gray[:, :-1, :]))
            return grad_x + grad_y
        
        # 2. Exposure Control Loss
        def exposure_control_loss(img, target=0.6, patch_size=16):
            # Sample patches
            B, C, H, W = img.shape
            patches = []
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    if i + patch_size <= H and j + patch_size <= W:
                        patch = img[:, :, i:i+patch_size, j:j+patch_size]
                        patches.append(torch.mean(patch))
            
            if patches:
                patches = torch.stack(patches)
                return torch.mean(torch.abs(patches - target))
            else:
                return torch.tensor(0.0, device=self.device)
        
        # 3. Color Constancy Loss
        def color_constancy_loss(img):
            mean_rgb = torch.mean(img, dim=[2, 3], keepdim=True)
            diff_r_g = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 1])
            diff_g_b = torch.abs(mean_rgb[:, 1] - mean_rgb[:, 2])
            diff_r_b = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 2])
            return torch.mean(diff_r_g) + torch.mean(diff_g_b) + torch.mean(diff_r_b)
        
        # 4. Illumination Smoothness Loss
        def illumination_smoothness_loss(curves):
            # Total variation loss for curve parameters
            tv_x = torch.mean(torch.abs(curves[:, :, :, 1:] - curves[:, :, :, :-1]))
            tv_y = torch.mean(torch.abs(curves[:, :, 1:, :] - curves[:, :, :-1, :]))
            return tv_x + tv_y
        
        # 5. Perceptual Loss (simplified)
        def perceptual_loss(enhanced, original):
            # Edge preservation
            gray_enh = 0.299 * enhanced[:, 0] + 0.587 * enhanced[:, 1] + 0.114 * enhanced[:, 2]
            gray_orig = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
            
            edge_enh = torch.abs(gray_enh[:, :, 1:] - gray_enh[:, :, :-1])
            edge_orig = torch.abs(gray_orig[:, :, 1:] - gray_orig[:, :, :-1])
            
            return torch.mean(torch.abs(edge_enh - edge_orig))
        
        # Calculate all losses
        loss_spa = spatial_consistency_loss(enhanced)
        loss_exp = exposure_control_loss(enhanced)
        loss_col = color_constancy_loss(enhanced)
        loss_tv = illumination_smoothness_loss(curves)
        loss_perc = perceptual_loss(enhanced, original)
        
        # Combined loss with weighted components
        total_loss = (loss_spa + 
                     10 * loss_exp + 
                     5 * loss_col + 
                     200 * loss_tv + 
                     0.1 * loss_perc)
        
        return total_loss, {
            'spatial': loss_spa.item(),
            'exposure': loss_exp.item(),
            'color': loss_col.item(),
            'smoothness': loss_tv.item(),
            'perceptual': loss_perc.item()
        }
    
    def train_epoch(self, dataloader, epoch, log_interval=10):
        """
        Train for one epoch with detailed logging
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (low_light, normal_light) in enumerate(pbar):
            low_light = low_light.to(self.device)
            normal_light = normal_light.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            curves = self.model(low_light)
            enhanced = self.model.apply_enhancement(low_light, curves)
            
            # Calculate loss
            loss, loss_components = self.zero_dce_loss(enhanced, low_light, curves)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Spa': f'{loss_components["spatial"]:.4f}',
                    'Exp': f'{loss_components["exposure"]:.4f}',
                    'Col': f'{loss_components["color"]:.4f}',
                    'TV': f'{loss_components["smoothness"]:.4f}'
                })
        
        avg_loss = epoch_loss / num_batches
        
        # Update scheduler
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        return avg_loss
    
    def validate(self, dataloader):
        """
        Validate the model
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for low_light, normal_light in tqdm(dataloader, desc="Validation"):
                low_light = low_light.to(self.device)
                normal_light = normal_light.to(self.device)
                
                curves = self.model(low_light)
                enhanced = self.model.apply_enhancement(low_light, curves)
                
                loss, _ = self.zero_dce_loss(enhanced, low_light, curves)
                val_loss += loss.item()
        
        return val_loss / num_batches
    
    def train(self, train_dataloader, val_dataloader=None, 
              num_epochs=200, save_dir="zerodce_checkpoints", 
              save_interval=20, early_stopping_patience=50):
        """
        Complete Zero-DCE training with early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"Starting Zero-DCE training for {num_epochs} epochs...")
        print(f"Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"Validation samples: {len(val_dataloader.dataset)}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    best_path = os.path.join(save_dir, "zerodce_best.pth")
                    self.save_model(best_path)
                    print(f"  🏆 New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save best based on training loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_path = os.path.join(save_dir, "zerodce_best.pth")
                    self.save_model(best_path)
                    print(f"  🏆 New best model saved! Train Loss: {train_loss:.6f}")
            
            # Save regular checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"zerodce_epoch_{epoch+1}.pth")
                self.save_model(checkpoint_path)
            
            # Record history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss if val_dataloader else None,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
        
        # Save final model
        final_path = os.path.join(save_dir, "zerodce_final.pth")
        self.save_model(final_path)
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n🎉 Zero-DCE training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Models saved in: {save_dir}")
        
        return self.training_history
    
    def save_model(self, path):
        """
        Save model checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'best_loss': self.best_loss
        }, path)
    
    def load_model(self, path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        print(f"Zero-DCE model loaded from: {path}")


class ZeroDCEDataset(Dataset):
    """
    Dataset for Zero-DCE training
    """
    
    def __init__(self, low_light_dir, normal_light_dir, transform=None, image_size=(400, 600)):
        self.low_light_images = []
        self.normal_light_images = []
        self.transform = transform
        self.image_size = image_size
        
        # Load low-light images
        self.low_light_images = sorted(glob.glob(os.path.join(low_light_dir, '*.*')))
        self.low_light_images = [f for f in self.low_light_images 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Load normal-light images
        self.normal_light_images = sorted(glob.glob(os.path.join(normal_light_dir, '*.*')))
        self.normal_light_images = [f for f in self.normal_light_images 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Loaded {len(self.low_light_images)} paired images")
    
    def __len__(self):
        return len(self.low_light_images)
    
    def __getitem__(self, idx):
        # Load images
        low_light = Image.open(self.low_light_images[idx]).convert('RGB')
        normal_light = Image.open(self.normal_light_images[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            low_light = self.transform(low_light)
            normal_light = self.transform(normal_light)
        
        return low_light, normal_light


def create_dataloaders(data_dir, batch_size=4, image_size=(400, 600)):
    """
    Create train and validation dataloaders for Zero-DCE
    """
    import torchvision.transforms as transforms
    from torch.utils.data import random_split
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    low_light_dir = os.path.join(data_dir, "low")
    normal_light_dir = os.path.join(data_dir, "high")
    
    dataset = ZeroDCEDataset(
        low_light_dir=low_light_dir,
        normal_light_dir=normal_light_dir,
        transform=transform,
        image_size=image_size
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """
    Main training function for Zero-DCE
    """
    print("=" * 60)
    print("ZERO-DCE COMPREHENSIVE TRAINING")
    print("=" * 60)
    
    # Configuration
    data_dir = "/home/sfm01/Downloads/luma/Lumanet-main/archive/lol_dataset/our485"
    save_dir = "zerodce_checkpoints"
    
    # Training parameters
    num_epochs = 200
    batch_size = 4
    learning_rate = 1e-4
    image_size = (400, 600)
    
    print(f"Configuration:")
    print(f"  Dataset: {data_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Image size: {image_size}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size, image_size)
    
    # Initialize trainer
    print("\nInitializing Zero-DCE trainer...")
    trainer = ZeroDCETrainer()
    trainer.setup_training(learning_rate=learning_rate, scheduler_type='cosine')
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    history = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=num_epochs,
        save_dir=save_dir,
        save_interval=20,
        early_stopping_patience=50
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best model saved to: {save_dir}/zerodce_best.pth")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()
