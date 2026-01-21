#!/usr/bin/env python3
"""
Quick verification script to test the setup.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from ota_fl import (
            generate_alpha_stable_noise,
            AlphaStableNoise,
            NoisyOracle,
            AdaGradOTA,
            AdamOTA,
            median_anchored_clipping,
        )
        from ota_fl.models import get_model
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_noise_generation():
    """Test noise generation."""
    print("\nTesting noise generation...")
    try:
        from ota_fl import generate_alpha_stable_noise
        
        # Test different alpha values
        for alpha in [1.5, 1.8, 2.0]:
            noise = generate_alpha_stable_noise(alpha=alpha, size=(1000,))
            print(f"  ✓ alpha={alpha}: mean={noise.mean():.4f}, std={noise.std():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Noise generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_noisy_oracle():
    """Test NoisyOracle."""
    print("\nTesting NoisyOracle...")
    try:
        from ota_fl import NoisyOracle
        
        oracle = NoisyOracle(alpha=1.5, noise_scale=0.1)
        
        # Simulate local updates from 10 clients
        local_updates = torch.randn(10, 100)
        aggregated = oracle.aggregate(local_updates)
        
        print(f"  ✓ Aggregation successful: input shape {local_updates.shape}, output shape {aggregated.shape}")
        return True
    except Exception as e:
        print(f"✗ NoisyOracle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizers():
    """Test optimizers."""
    print("\nTesting optimizers...")
    try:
        from ota_fl import AdaGradOTA, AdamOTA
        from ota_fl.models import get_model
        
        model = get_model("mlp", num_classes=10)
        
        # Test AdaGrad-OTA
        opt1 = AdaGradOTA(model.parameters(), lr=0.01, alpha=1.5)
        print("  ✓ AdaGrad-OTA initialized")
        
        # Test Adam-OTA
        opt2 = AdamOTA(model.parameters(), lr=0.001, alpha=1.5)
        print("  ✓ Adam-OTA initialized")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 50)
    print("OTA-FL Setup Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_noise_generation,
        test_noisy_oracle,
        test_optimizers,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
