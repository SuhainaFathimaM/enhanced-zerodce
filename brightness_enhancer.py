#!/usr/bin/env python3
"""
Advanced Brightness Enhancement for Enhanced Zero-DCE
Multiple brightness enhancement strategies
"""

import torch
import torch.nn.functional as F
import numpy as np

class AdvancedBrightnessEnhancer:
    """
    Advanced brightness enhancement with multiple strategies
    """
    
    def __init__(self):
        self.strategies = {
            'adaptive': self.adaptive_brightness_enhancement,
            'gamma': self.gamma_correction_enhancement,
            'histogram': self.histogram_equalization_enhancement,
            'multi_scale': self.multi_scale_brightness,
            'aggressive': self.aggressive_brightness_boost
        }
    
    def adaptive_brightness_enhancement(self, image_tensor):
        """
        Adaptive brightness enhancement based on image statistics
        """
        # Calculate current brightness
        current_brightness = torch.mean(image_tensor, dim=[1, 2, 3], keepdim=True)
        
        # Target brightness levels for different scenarios
        if current_brightness.mean() < 0.1:  # Very dark
            target_brightness = 0.7
            max_boost = 6.0
        elif current_brightness.mean() < 0.2:  # Dark
            target_brightness = 0.6
            max_boost = 4.0
        elif current_brightness.mean() < 0.3:  # Moderately dark
            target_brightness = 0.5
            max_boost = 3.0
        else:  # Normal or bright
            target_brightness = 0.4
            max_boost = 2.0
        
        # Calculate adaptive boost
        brightness_boost = target_brightness / (current_brightness.mean() + 1e-8)
        brightness_boost = torch.clamp(brightness_boost, 1.0, max_boost)
        
        # Apply enhancement
        enhanced = image_tensor * brightness_boost
        
        # Apply gentle gamma correction
        gamma = 0.85 if current_brightness.mean() < 0.2 else 0.9
        enhanced = torch.pow(enhanced + 1e-8, gamma)
        
        return torch.clamp(enhanced, 0, 1)
    
    def gamma_correction_enhancement(self, image_tensor):
        """
        Gamma correction for brightness enhancement
        """
        # Adaptive gamma based on brightness
        brightness = torch.mean(image_tensor)
        
        if brightness < 0.1:
            gamma = 0.6  # Strong brightening
        elif brightness < 0.2:
            gamma = 0.7  # Moderate brightening
        elif brightness < 0.3:
            gamma = 0.8  # Light brightening
        else:
            gamma = 0.9  # Very light brightening
        
        enhanced = torch.pow(image_tensor + 1e-8, gamma)
        return torch.clamp(enhanced, 0, 1)
    
    def histogram_equalization_enhancement(self, image_tensor):
        """
        Histogram equalization style enhancement
        """
        # Convert to numpy for histogram processing
        img_np = image_tensor.squeeze(0).cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Apply CLAHE-like enhancement
        enhanced_np = np.zeros_like(img_np)
        
        for c in range(3):
            channel = img_np[:, :, c]
            
            # Simple histogram stretching
            min_val = np.percentile(channel, 2)
            max_val = np.percentile(channel, 98)
            
            if max_val > min_val:
                stretched = (channel - min_val) / (max_val - min_val)
                enhanced_np[:, :, c] = np.clip(stretched, 0, 1)
            else:
                enhanced_np[:, :, c] = channel
        
        # Convert back to tensor
        enhanced_np = np.transpose(enhanced_np, (2, 0, 1))
        enhanced_tensor = torch.from_numpy(enhanced_np).unsqueeze(0).to(image_tensor.device)
        
        return enhanced_tensor
    
    def multi_scale_brightness(self, image_tensor):
        """
        Multi-scale brightness enhancement
        """
        enhanced = image_tensor.clone()
        
        # Apply enhancement at different scales
        scales = [1.0, 0.5, 0.25]
        
        for scale in scales:
            if scale < 1.0:
                # Downsample
                h, w = image_tensor.shape[2], image_tensor.shape[3]
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                scaled = F.interpolate(image_tensor, size=(scaled_h, scaled_w), mode='bilinear')
                
                # Enhance brightness
                brightness = torch.mean(scaled)
                if brightness < 0.3:
                    boost = 1.5 / (brightness + 1e-8)
                    boost = torch.clamp(boost, 1.0, 3.0)
                    scaled = scaled * boost
                
                # Upsample back
                scaled_up = F.interpolate(scaled, size=(h, w), mode='bilinear')
                enhanced = 0.7 * enhanced + 0.3 * scaled_up
        
        return torch.clamp(enhanced, 0, 1)
    
    def aggressive_brightness_boost(self, image_tensor):
        """
        Aggressive brightness boost for very dark images
        """
        brightness = torch.mean(image_tensor)
        
        if brightness < 0.05:  # Extremely dark
            # Multiple enhancement stages
            enhanced = image_tensor.clone()
            
            # Stage 1: Linear boost
            enhanced = enhanced * 4.0
            
            # Stage 2: Gamma correction
            enhanced = torch.pow(enhanced + 1e-8, 0.6)
            
            # Stage 3: Adaptive boost
            current_brightness = torch.mean(enhanced)
            if current_brightness < 0.4:
                final_boost = 0.5 / (current_brightness + 1e-8)
                final_boost = torch.clamp(final_boost, 1.0, 2.5)
                enhanced = enhanced * final_boost
            
            return torch.clamp(enhanced, 0, 1)
        
        else:
            # Use adaptive enhancement for less dark images
            return self.adaptive_brightness_enhancement(image_tensor)
    
    def enhance(self, image_tensor, strategy='adaptive'):
        """
        Enhance image brightness using specified strategy
        """
        if strategy not in self.strategies:
            print(f"Strategy '{strategy}' not found. Using 'adaptive'")
            strategy = 'adaptive'
        
        return self.strategies[strategy](image_tensor)
    
    def compare_strategies(self, image_tensor):
        """
        Compare all brightness enhancement strategies
        """
        print("🔬 Comparing Brightness Enhancement Strategies")
        print("=" * 50)
        
        original_brightness = torch.mean(image_tensor).item()
        print(f"Original brightness: {original_brightness:.4f}")
        print()
        
        results = {}
        
        for strategy_name, strategy_func in self.strategies.items():
            try:
                enhanced = strategy_func(image_tensor.clone())
                enhanced_brightness = torch.mean(enhanced).item()
                improvement = ((enhanced_brightness - original_brightness) / original_brightness) * 100
                
                results[strategy_name] = {
                    'brightness': enhanced_brightness,
                    'improvement_percent': improvement
                }
                
                print(f"{strategy_name:12}: {enhanced_brightness:8.4f} ({improvement:+6.1f}%)")
                
            except Exception as e:
                print(f"{strategy_name:12}: ERROR - {e}")
        
        # Find best strategy
        if results:
            best_strategy = max(results.items(), key=lambda x: x[1]['brightness'])
            print(f"\n🏆 Best strategy: {best_strategy[0]} (brightness: {best_strategy[1]['brightness']:.4f})")
        
        return results


def test_brightness_enhancement():
    """
    Test the advanced brightness enhancement
    """
    enhancer = AdvancedBrightnessEnhancer()
    
    # Create test images with different brightness levels
    test_cases = [
        ("Very Dark", np.random.randint(0, 10, (400, 600, 3), dtype=np.uint8)),
        ("Dark", np.random.randint(10, 30, (400, 600, 3), dtype=np.uint8)),
        ("Moderate", np.random.randint(30, 60, (400, 600, 3), dtype=np.uint8))
    ]
    
    for name, test_image in test_cases:
        print(f"\n🧪 Testing {name} Image:")
        print("-" * 30)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(test_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Compare strategies
        results = enhancer.compare_strategies(image_tensor)
        print()


if __name__ == "__main__":
    test_brightness_enhancement()
