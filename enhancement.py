import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import enhanced Zero-DCE only
try:
    from enhanced_zerodce import EnhancedZeroDCEEnhancer
    ENHANCED_ZERODCE_AVAILABLE = True
except ImportError:
    ENHANCED_ZERODCE_AVAILABLE = False
    print("Enhanced Zero-DCE not available.")

class LowLightEnhancer:
    def __init__(self):
        self.enhanced_zerodce = None
        
        # Load enhanced Zero-DCE with innovations
        if ENHANCED_ZERODCE_AVAILABLE:
            try:
                self.enhanced_zerodce = EnhancedZeroDCEEnhancer(use_enhanced=True)
                self.enhanced_zerodce.load_model()
                print("🚀 Enhanced Zero-DCE with innovations loaded successfully!")
            except Exception as e:
                print(f"Failed to load enhanced Zero-DCE: {e}")
                self.enhanced_zerodce = None
    
    def enhance_image(self, image, method='enhanced_zerodce'):
        """
        Enhance low-light image using enhanced Zero-DCE
        
        Args:
            image: Input image (BGR format)
            method: Enhancement method ('enhanced_zerodce')
        
        Returns:
            Enhanced image (BGR format)
        """
        if method == 'enhanced_zerodce':
            return self.enhance_enhanced_zerodce(image)
        else:
            return self.enhance_enhanced_zerodce(image)
    
    def enhance_enhanced_zerodce(self, image):
        """Enhanced Zero-DCE with all innovations"""
        if self.enhanced_zerodce is None:
            print("Enhanced Zero-DCE not available, using basic enhancement")
            return self.clahe_enhancement(image)
        
        try:
            return self.enhanced_zerodce.enhance_image(image)
        except Exception as e:
            print(f"Enhanced Zero-DCE failed: {e}, falling back to CLAHE")
            return self.clahe_enhancement(image)
    
    def clahe_enhancement(self, image):
        """Basic CLAHE enhancement as fallback"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def get_available_methods(self):
        """Get available enhancement methods"""
        methods = ['enhanced_zerodce']
        return methods
