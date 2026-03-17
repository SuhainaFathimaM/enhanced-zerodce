#!/usr/bin/env python3
"""
Enhanced Zero-DCE Model Selector
Choose which trained model to use from enhanced_zerodce directory
"""

import os
import torch
from enhanced_zerodce import EnhancedZeroDCEEnhancer
import json

class EnhancedModelSelector:
    """
    Select and test different trained models from enhanced_zerodce directory
    """
    
    def __init__(self):
        self.enhanced_dir = "enhanced_zerodce"
        self.available_models = self.scan_available_models()
        
    def scan_available_models(self):
        """Scan for available trained models"""
        models = {}
        
        if os.path.exists(self.enhanced_dir):
            for file in os.listdir(self.enhanced_dir):
                if file.endswith('.pth'):
                    model_name = file.replace('.pth', '').replace('enhanced_zerodce_', '')
                    model_path = os.path.join(self.enhanced_dir, file)
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    
                    models[model_name] = {
                        'path': model_path,
                        'size_mb': file_size,
                        'full_name': file
                    }
        
        return models
    
    def list_models(self):
        """List all available models"""
        print("🚀 Enhanced Zero-DCE Available Models:")
        print("=" * 50)
        
        for i, (name, info) in enumerate(self.available_models.items(), 1):
            print(f"{i}. {name}")
            print(f"   Path: {info['path']}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print()
        
        return list(self.available_models.keys())
    
    def test_model(self, model_name):
        """Test a specific model"""
        if model_name not in self.available_models:
            print(f"❌ Model '{model_name}' not found")
            return None
        
        model_path = self.available_models[model_name]['path']
        print(f"🧪 Testing model: {model_name}")
        print(f"📁 Path: {model_path}")
        
        try:
            # Load enhancer with specific model
            enhancer = EnhancedZeroDCEEnhancer(
                model_path=model_path,
                use_enhanced=True
            )
            
            success = enhancer.load_model()
            if not success:
                print("❌ Failed to load model")
                return None
            
            # Test with sample image
            import numpy as np
            test_image = np.random.randint(0, 30, (400, 600, 3), dtype=np.uint8)
            
            result = enhancer.enhance_image(test_image)
            
            original_brightness = np.mean(test_image)
            enhanced_brightness = np.mean(result)
            improvement = ((enhanced_brightness - original_brightness) / original_brightness) * 100
            
            print(f"✅ Model tested successfully!")
            print(f"📊 Original brightness: {original_brightness:.2f}")
            print(f"📊 Enhanced brightness: {enhanced_brightness:.2f}")
            print(f"📈 Improvement: {improvement:.1f}%")
            
            return {
                'model_name': model_name,
                'path': model_path,
                'original_brightness': original_brightness,
                'enhanced_brightness': enhanced_brightness,
                'improvement_percent': improvement,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return {
                'model_name': model_name,
                'path': model_path,
                'error': str(e),
                'success': False
            }
    
    def compare_all_models(self):
        """Compare all available models"""
        print("🔬 Comparing All Enhanced Zero-DCE Models")
        print("=" * 60)
        
        results = []
        
        for model_name in self.available_models.keys():
            result = self.test_model(model_name)
            if result:
                results.append(result)
            print("-" * 40)
        
        # Find best performing model
        successful_results = [r for r in results if r.get('success', False)]
        
        if successful_results:
            best_model = max(successful_results, key=lambda x: x['improvement_percent'])
            print(f"\n🏆 BEST PERFORMING MODEL:")
            print(f"   Model: {best_model['model_name']}")
            print(f"   Improvement: {best_model['improvement_percent']:.1f}%")
            print(f"   Path: {best_model['path']}")
        
        return results
    
    def get_training_info(self):
        """Get training information from history"""
        history_path = os.path.join(self.enhanced_dir, "enhanced_training_history.json")
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                print("📊 Training Information:")
                print(f"   Total Epochs: {len(history)}")
                
                if history:
                    first_epoch = history[0]
                    last_epoch = history[-1]
                    
                    print(f"   Initial Loss: {first_epoch.get('train_loss', 'N/A'):.4f}")
                    print(f"   Final Loss: {last_epoch.get('train_loss', 'N/A'):.4f}")
                    print(f"   Best Val Loss: {min([h.get('val_loss', float('inf')) for h in history if h.get('val_loss')]):.4f}")
                    print(f"   Final Learning Rate: {last_epoch.get('learning_rate', 'N/A'):.2e}")
                
                return history
                
            except Exception as e:
                print(f"❌ Failed to read training history: {e}")
                return None
        else:
            print("❌ No training history found")
            return None


def main():
    """Main function for model selection"""
    selector = EnhancedModelSelector()
    
    print("🚀 Enhanced Zero-DCE Model Selector")
    print("=" * 50)
    
    # List available models
    models = selector.list_models()
    
    if not models:
        print("❌ No models found in enhanced_zerodce directory")
        return
    
    # Get training info
    selector.get_training_info()
    
    print("\n" + "=" * 50)
    print("Choose an option:")
    print("1. Test a specific model")
    print("2. Compare all models")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        model_name = input(f"Enter model name ({', '.join(models)}): ").strip()
        selector.test_model(model_name)
    
    elif choice == "2":
        selector.compare_all_models()
    
    elif choice == "3":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
