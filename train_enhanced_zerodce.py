#!/usr/bin/env python3
"""
Enhanced Zero-DCE Training Script
Trains your innovative Zero-DCE model with all improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json

from enhanced_zerodce import EnhancedZeroDCENet
from zerodce_trainer import ZeroDCEDataset, ZeroDCETrainer

class EnhancedZeroDCETrainer(ZeroDCETrainer):
    """
    Trainer for Enhanced Zero-DCE with all innovations
    """
    
    def __init__(self, model=None, device='auto'):
        if model is None:
            device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
            model = EnhancedZeroDCENet(iteration=8, use_attention=True, multi_scale=True).to(device)
        else:
            device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
        super().__init__(model, device)
        
        print(f"Enhanced Zero-DCE Trainer initialized")
        print(f"Innovations: Multi-scale, Attention, Adaptive Iteration, Residual")
    
    def enhanced_loss(self, enhanced, original, curves):
        """
        Enhanced loss function with all components
        """
        # 1. Spatial Consistency Loss
        def spatial_consistency_loss(img):
            gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
            grad_x = torch.mean(torch.abs(gray[:, :, 1:] - gray[:, :, :-1]))
            grad_y = torch.mean(torch.abs(gray[:, 1:, :] - gray[:, :-1, :]))
            return grad_x + grad_y
        
        # 2. Exposure Control Loss (enhanced)
        def exposure_control_loss(img, target=0.6, patch_size=16):
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
            return torch.tensor(0.0, device=self.device)
        
        # 3. Color Constancy Loss (enhanced)
        def color_constancy_loss(img):
            mean_rgb = torch.mean(img, dim=[2, 3], keepdim=True)
            diff_r_g = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 1])
            diff_g_b = torch.abs(mean_rgb[:, 1] - mean_rgb[:, 2])
            diff_r_b = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 2])
            return torch.mean(diff_r_g) + torch.mean(diff_g_b) + torch.mean(diff_r_b)
        
        # 4. Illumination Smoothness Loss (enhanced)
        def illumination_smoothness_loss(curves):
            tv_x = torch.mean(torch.abs(curves[:, :, :, 1:] - curves[:, :, :, :-1]))
            tv_y = torch.mean(torch.abs(curves[:, :, 1:, :] - curves[:, :, :-1, :]))
            return tv_x + tv_y
        
        # 5. Perceptual Loss (NEW)
        def perceptual_loss(enhanced, original):
            gray_enh = 0.299 * enhanced[:, 0] + 0.587 * enhanced[:, 1] + 0.114 * enhanced[:, 2]
            gray_orig = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, 2]
            
            edge_enh = torch.abs(gray_enh[:, :, 1:] - gray_enh[:, :, :-1])
            edge_orig = torch.abs(gray_orig[:, :, 1:] - gray_orig[:, :, :-1])
            
            return torch.mean(torch.abs(edge_enh - edge_orig))
        
        # 6. Multi-scale Consistency Loss (NEW)
        def multiscale_loss(img):
            loss = 0.0
            for scale in [0.5, 0.25]:
                h, w = img.shape[2], img.shape[3]
                scaled = F.interpolate(img, scale_factor=scale, mode='bilinear')
                gray = 0.299 * scaled[:, 0] + 0.587 * scaled[:, 1] + 0.114 * scaled[:, 2]
                grad = torch.mean(torch.abs(gray[:, :, 1:] - gray[:, :, :-1]))
                loss += grad
            return loss
        
        # Calculate all losses
        loss_spa = spatial_consistency_loss(enhanced)
        loss_exp = exposure_control_loss(enhanced)
        loss_col = color_constancy_loss(enhanced)
        loss_tv = illumination_smoothness_loss(curves)
        loss_perc = perceptual_loss(enhanced, original)
        loss_ms = multiscale_loss(enhanced)
        
        # Enhanced combined loss
        total_loss = (loss_spa + 
                     15 * loss_exp +      # Increased exposure weight
                     8 * loss_col +       # Increased color weight  
                     300 * loss_tv +      # Increased smoothness weight
                     0.2 * loss_perc +    # Perceptual loss
                     0.5 * loss_ms)       # Multi-scale loss
        
        return total_loss, {
            'spatial': loss_spa.item(),
            'exposure': loss_exp.item(),
            'color': loss_col.item(),
            'smoothness': loss_tv.item(),
            'perceptual': loss_perc.item(),
            'multiscale': loss_ms.item()
        }
    
    def train_epoch(self, dataloader, epoch, log_interval=10):
        """
        Enhanced training epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Enhanced Epoch {epoch+1}")
        
        for batch_idx, (low_light, normal_light) in enumerate(pbar):
            low_light = low_light.to(self.device)
            normal_light = normal_light.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with enhanced model
            curves = self.model(low_light)
            enhanced = self.model.apply_enhancement(low_light, curves)
            
            # Calculate enhanced loss
            loss, loss_components = self.enhanced_loss(enhanced, low_light, curves)
            
            # Backward pass
            loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Spa': f'{loss_components["spatial"]:.4f}',
                    'Exp': f'{loss_components["exposure"]:.4f}',
                    'Col': f'{loss_components["color"]:.4f}',
                    'TV': f'{loss_components["smoothness"]:.4f}',
                    'Perc': f'{loss_components["perceptual"]:.4f}'
                })
        
        avg_loss = epoch_loss / num_batches
        
        # Update scheduler
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, 
              num_epochs=100, save_dir="enhanced_zerodce", 
              save_interval=20, early_stopping_patience=30):
        """
        Enhanced training with all innovations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"🚀 Starting Enhanced Zero-DCE Training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"📊 Validation samples: {len(val_dataloader.dataset)}")
        
        print(f"🔧 Model Innovations:")
        print(f"   ✅ Multi-scale feature extraction")
        print(f"   ✅ Self-attention mechanism")  
        print(f"   ✅ Adaptive curve iteration")
        print(f"   ✅ Residual connections")
        print(f"   ✅ Enhanced loss functions")
        
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
                    best_path = os.path.join(save_dir, "enhanced_zerodce_best.pth")
                    self.save_model(best_path)
                    print(f"  🏆 New best enhanced model! Val Loss: {val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_path = os.path.join(save_dir, "enhanced_zerodce_best.pth")
                    self.save_model(best_path)
                    print(f"  🏆 New best enhanced model! Train Loss: {train_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"enhanced_zerodce_epoch_{epoch+1}.pth")
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
        final_path = os.path.join(save_dir, "enhanced_zerodce_final.pth")
        self.save_model(final_path)
        
        # Save training history
        history_path = os.path.join(save_dir, "enhanced_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n🎉 Enhanced Zero-DCE Training Completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.6f}")
        print(f"💾 Models saved in: {save_dir}")
        
        return self.training_history


def main():
    """
    Main enhanced training function
    """
    print("=" * 70)
    print("🚀 ENHANCED ZERO-DCE TRAINING WITH INNOVATIONS")
    print("=" * 70)
    
    # Configuration
    data_dir = "/home/sfm01/Downloads/luma/Lumanet-main/archive/lol_dataset/our485"
    save_dir = "enhanced_zerodce"
    
    # Training parameters
    num_epochs = 80  # Focused training with innovations
    batch_size = 2
    learning_rate = 5e-5  # Lower LR for stable training
    image_size = (256, 256)  # Smaller for faster training
    
    print(f"📋 Configuration:")
    print(f"   Dataset: {data_dir}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Image size: {image_size}")
    
    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    from zerodce_trainer import create_dataloaders
    train_loader, val_loader = create_dataloaders(data_dir, batch_size, image_size)
    
    # Initialize enhanced trainer
    print(f"\n🤖 Initializing Enhanced Zero-DCE Trainer...")
    trainer = EnhancedZeroDCETrainer()
    trainer.setup_training(learning_rate=learning_rate, scheduler_type='cosine')
    
    # Train enhanced model
    print(f"\n🏃‍♂️ Starting Enhanced Training...")
    start_time = time.time()
    history = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=num_epochs,
        save_dir=save_dir,
        save_interval=20,
        early_stopping_patience=30
    )
    training_time = time.time() - start_time
    
    print(f"\n⏱️  Enhanced training completed in {training_time/3600:.2f} hours")
    print(f"🏆 Best enhanced model: {save_dir}/enhanced_zerodce_best.pth")
    
    return trainer, history


if __name__ == "__main__":
    trainer, history = main()
