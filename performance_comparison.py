#!/usr/bin/env python3
"""
Performance Comparison: Enhanced Zero-DCE vs Original Zero-DCE
Tests if your innovations actually improve performance
"""

import torch
import cv2
import numpy as np
import time
import os
from pathlib import Path
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import models
from enhanced_zerodce import EnhancedZeroDCEEnhancer
from zero_dce_model import ZeroDCEEnhancer

class PerformanceComparator:
    """
    Compare Enhanced Zero-DCE vs Original Zero-DCE
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Performance Comparator on {self.device}")
        
        # Load models
        self.enhanced_enhancer = None
        self.original_enhancer = None
        
        self.load_models()
    
    def load_models(self):
        """Load both models for comparison"""
        try:
            print("Loading Enhanced Zero-DCE...")
            self.enhanced_enhancer = EnhancedZeroDCEEnhancer(use_enhanced=True, device=self.device)
            self.enhanced_enhancer.load_model()
            print("✅ Enhanced Zero-DCE loaded")
        except Exception as e:
            print(f"❌ Enhanced Zero-DCE failed: {e}")
        
        try:
            print("Loading Original Zero-DCE...")
            self.original_enhancer = ZeroDCEEnhancer(device=self.device)
            self.original_enhancer.load_model()
            print("✅ Original Zero-DCE loaded")
        except Exception as e:
            print(f"❌ Original Zero-DCE failed: {e}")
    
    def calculate_metrics(self, original, enhanced, reference=None):
        """Calculate image quality metrics"""
        # Convert to grayscale for SSIM
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        ssim_val = ssim(orig_gray, enh_gray, data_range=255)
        psnr_val = psnr(orig_gray, enh_gray, data_range=255)
        
        # Brightness improvement
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)
        brightness_improvement = (enh_brightness - orig_brightness) / orig_brightness * 100
        
        # Contrast improvement
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100
        
        return {
            'ssim': ssim_val,
            'psnr': psnr_val,
            'brightness_improvement': brightness_improvement,
            'contrast_improvement': contrast_improvement,
            'original_brightness': orig_brightness,
            'enhanced_brightness': enh_brightness
        }
    
    def test_single_image(self, image_path):
        """Test both models on a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        print(f"\nTesting: {os.path.basename(image_path)}")
        print(f"Original size: {image.shape}")
        
        results = {}
        
        # Test Enhanced Zero-DCE
        if self.enhanced_enhancer:
            try:
                start_time = time.time()
                enhanced_result = self.enhanced_enhancer.enhance_image(image)
                enhanced_time = time.time() - start_time
                
                enhanced_metrics = self.calculate_metrics(image, enhanced_result)
                results['enhanced'] = {
                    'time': enhanced_time,
                    'metrics': enhanced_metrics,
                    'success': True
                }
                print(f"✅ Enhanced: {enhanced_time:.3f}s, SSIM: {enhanced_metrics['ssim']:.4f}")
            except Exception as e:
                results['enhanced'] = {'success': False, 'error': str(e)}
                print(f"❌ Enhanced failed: {e}")
        
        # Test Original Zero-DCE
        if self.original_enhancer:
            try:
                start_time = time.time()
                original_result = self.original_enhancer.enhance_image(image)
                original_time = time.time() - start_time
                
                original_metrics = self.calculate_metrics(image, original_result)
                results['original'] = {
                    'time': original_time,
                    'metrics': original_metrics,
                    'success': True
                }
                print(f"✅ Original: {original_time:.3f}s, SSIM: {original_metrics['ssim']:.4f}")
            except Exception as e:
                results['original'] = {'success': False, 'error': str(e)}
                print(f"❌ Original failed: {e}")
        
        return results, image, enhanced_result if 'enhanced_result' in locals() else None, original_result if 'original_result' in locals() else None
    
    def compare_models(self, test_images):
        """Compare models on multiple test images"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        all_results = []
        
        for img_path in test_images:
            results, orig_img, enh_result, orig_result = self.test_single_image(img_path)
            
            if results:
                comparison = {
                    'image': os.path.basename(img_path),
                    'results': results
                }
                
                # Calculate improvement if both succeeded
                if results.get('enhanced', {}).get('success') and results.get('original', {}).get('success'):
                    enh_metrics = results['enhanced']['metrics']
                    orig_metrics = results['original']['metrics']
                    
                    comparison['improvement'] = {
                        'ssim': enh_metrics['ssim'] - orig_metrics['ssim'],
                        'psnr': enh_metrics['psnr'] - orig_metrics['psnr'],
                        'brightness': enh_metrics['brightness_improvement'] - orig_metrics['brightness_improvement'],
                        'contrast': enh_metrics['contrast_improvement'] - orig_metrics['contrast_improvement'],
                        'time': results['original']['time'] - results['enhanced']['time']
                    }
                    
                    print(f"📊 SSIM improvement: {comparison['improvement']['ssim']:+.4f}")
                    print(f"📊 PSNR improvement: {comparison['improvement']['psnr']:+.2f}")
                    print(f"📊 Speed improvement: {comparison['improvement']['time']:+.3f}s")
                
                all_results.append(comparison)
        
        # Calculate overall statistics
        self.calculate_overall_stats(all_results)
        
        return all_results
    
    def calculate_overall_stats(self, results):
        """Calculate overall performance statistics"""
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE ANALYSIS")
        print("="*60)
        
        successful_comparisons = [r for r in results if 'improvement' in r]
        
        if not successful_comparisons:
            print("❌ No successful comparisons to analyze")
            return
        
        # Calculate average improvements
        improvements = [r['improvement'] for r in successful_comparisons]
        
        avg_ssim = np.mean([imp['ssim'] for imp in improvements])
        avg_psnr = np.mean([imp['psnr'] for imp in improvements])
        avg_brightness = np.mean([imp['brightness'] for imp in improvements])
        avg_contrast = np.mean([imp['contrast'] for imp in improvements])
        avg_time = np.mean([imp['time'] for imp in improvements])
        
        print(f"\n📈 AVERAGE IMPROVEMENTS:")
        print(f"   SSIM: {avg_ssim:+.4f} ({'BETTER' if avg_ssim > 0 else 'WORSE'})")
        print(f"   PSNR: {avg_psnr:+.2f} dB ({'BETTER' if avg_psnr > 0 else 'WORSE'})")
        print(f"   Brightness: {avg_brightness:+.1f}% ({'BETTER' if avg_brightness > 0 else 'WORSE'})")
        print(f"   Contrast: {avg_contrast:+.1f}% ({'BETTER' if avg_contrast > 0 else 'WORSE'})")
        print(f"   Speed: {avg_time:+.3f}s ({'FASTER' if avg_time > 0 else 'SLOWER'})")
        
        # Determine overall winner
        better_metrics = sum([
            avg_ssim > 0,
            avg_psnr > 0,
            avg_brightness > 0,
            avg_contrast > 0,
            avg_time > 0
        ])
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if better_metrics >= 3:
            print("   ✅ ENHANCED ZERO-DCE PERFORMS BETTER!")
            print(f"   ✅ Better in {better_metrics}/5 metrics")
        elif better_metrics <= 1:
            print("   ❌ ENHANCED ZERO-DCE PERFORMS WORSE!")
            print("   ⚠️  Your innovations may need refinement")
        else:
            print("   ⚖️  MIXED RESULTS - Some improvements, some regressions")
            print("   🔧 Consider optimizing specific components")
        
        # Save results
        self.save_results(results, {
            'average_improvements': {
                'ssim': avg_ssim,
                'psnr': avg_psnr,
                'brightness': avg_brightness,
                'contrast': avg_contrast,
                'time': avg_time
            },
            'better_metrics_count': better_metrics,
            'overall_winner': 'enhanced' if better_metrics >= 3 else 'original'
        })
    
    def save_results(self, detailed_results, summary):
        """Save comparison results"""
        results = {
            'summary': summary,
            'detailed_results': detailed_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('performance_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: performance_comparison_results.json")
    
    def get_test_images(self):
        """Get test images for comparison"""
        # Look for test images in common locations
        test_paths = [
            "/home/sfm01/Downloads/luma/Lumanet-main/archive/lol_dataset/our485/low",
            "uploads",
            "."
        ]
        
        test_images = []
        for path in test_paths:
            if os.path.exists(path):
                images = [f for f in os.listdir(path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
                for img in images:
                    test_images.append(os.path.join(path, img))
                if test_images:
                    break
        
        return test_images


def main():
    """Main comparison function"""
    print("🔬 ENHANCED ZERO-DCE vs ORIGINAL ZERO-DCE COMPARISON")
    print("="*60)
    
    comparator = PerformanceComparator()
    test_images = comparator.get_test_images()
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Run comparison
    results = comparator.compare_models(test_images)
    
    return results


if __name__ == "__main__":
    results = main()
