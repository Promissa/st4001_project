"""
Main entry point for OTA-FL experiments.

For training, use: python -m src.experiments.train --config configs/default.yaml
"""

def main():
    print("Adaptive Federated Learning Over the Air")
    print("=" * 50)
    print("To run experiments, use:")
    print("  python -m src.experiments.train --config configs/default.yaml")
    print("\nTo verify setup, use:")
    print("  python verify_setup.py")


if __name__ == "__main__":
    main()
