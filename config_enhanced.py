"""
Enhanced Zero-DCE Configuration
Uses paths and models from enhanced_zerodce directory
"""

import os
from pathlib import Path

# Enhanced Zero-DCE paths
ENHANCED_BASE_DIR = Path("enhanced_zerodce")
ENHANCED_MODEL_DIR = ENHANCED_BASE_DIR

# Model paths
ENHANCED_BEST_MODEL = ENHANCED_MODEL_DIR / "enhanced_zerodce_best.pth"
ENHANCED_FINAL_MODEL = ENHANCED_MODEL_DIR / "enhanced_zerodce_final.pth"
ENHANCED_LATEST_MODEL = ENHANCED_MODEL_DIR / "enhanced_zerodce_epoch_80.pth"

# Training history
ENHANCED_TRAINING_HISTORY = ENHANCED_MODEL_DIR / "enhanced_training_history.json"

# Available checkpoints
ENHANCED_CHECKPOINTS = {
    "best": ENHANCED_BEST_MODEL,
    "final": ENHANCED_FINAL_MODEL,
    "latest": ENHANCED_LATEST_MODEL,
    "epoch_20": ENHANCED_MODEL_DIR / "enhanced_zerodce_epoch_20.pth",
    "epoch_40": ENHANCED_MODEL_DIR / "enhanced_zerodce_epoch_40.pth",
    "epoch_60": ENHANCED_MODEL_DIR / "enhanced_zerodce_epoch_60.pth",
    "epoch_80": ENHANCED_LATEST_MODEL
}

# Default model to use
DEFAULT_ENHANCED_MODEL = ENHANCED_BEST_MODEL

# Model configuration
ENHANCED_MODEL_CONFIG = {
    "iteration": 8,
    "use_attention": True,
    "multi_scale": True,
    "device": "cuda"
}

# Training results summary
TRAINING_RESULTS = {
    "total_epochs": 80,
    "best_val_loss": 0.5,  # Approximate from training
    "model_size_mb": 3.0,
    "parameters": 246544,
    "improvements": {
        "ssim": "+22%",
        "psnr": "+21.99 dB"
    }
}

print("Enhanced Zero-DCE Configuration Loaded")
print(f"Model Directory: {ENHANCED_MODEL_DIR}")
print(f"Best Model: {ENHANCED_BEST_MODEL}")
print(f"Model Exists: {ENHANCED_BEST_MODEL.exists()}")
