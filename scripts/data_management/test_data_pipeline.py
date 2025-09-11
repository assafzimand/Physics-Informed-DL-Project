"""
Test script for Wave Dataset Pipeline

Tests data loading, normalization, and DataLoader functionality.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.wave_dataset import WaveDataset, create_dataloaders, inspect_dataset


def test_single_dataset():
    """Test loading from a single dataset file."""
    print("=" * 60)
    print("Testing Single Dataset Loading")
    print("=" * 60)

    dataset_path = "data/wave_dataset_T250.h5"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please generate datasets first using scripts/generate_full_datasets.py")
        return False

    try:
        # Test dataset creation
        dataset = WaveDataset(dataset_path)
        print("✓ Dataset created successfully")
        print(f"✓ Dataset size: {len(dataset)}")

        # Test sample loading
        wave_field, coordinates = dataset[0]
        print("✓ Sample loading works")
        print(f"  - Wave field shape: {wave_field.shape}")
        print(f"  - Coordinates shape: {coordinates.shape}")
        print(f"  - Wave field range: [{wave_field.min():.4f}, {wave_field.max():.4f}]")
        print(f"  - Coordinates: ({coordinates[0]:.1f}, {coordinates[1]:.1f})")

        return True

    except Exception as e:
        print(f"❌ Error testing single dataset: {e}")
        return False


def test_dataloader_creation():
    """Test DataLoader creation and batching."""
    print("\n" + "=" * 60)
    print("Testing DataLoader Creation")
    print("=" * 60)

    dataset_path = "data/wave_dataset_T250.h5"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    try:
        # Create DataLoaders
        train_loader, val_loader = create_dataloaders(
            dataset_path,
            batch_size=8,
            validation_split=0.2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        print("✓ DataLoaders created successfully")
        print(f"✓ Training batches: {len(train_loader)}")
        print(f"✓ Validation batches: {len(val_loader)}")

        # Test batch loading
        train_batch = next(iter(train_loader))
        wave_fields, coordinates = train_batch

        print("✓ Batch loading works")
        print(f"  - Wave fields batch shape: {wave_fields.shape}")
        print(f"  - Coordinates batch shape: {coordinates.shape}")
        print(
            f"  - Wave fields range: [{wave_fields.min():.4f}, {wave_fields.max():.4f}]"
        )
        print(
            f"  - Coordinates range: [{coordinates.min():.1f}, {coordinates.max():.1f}]"
        )

        return True

    except Exception as e:
        print(f"❌ Error testing DataLoader: {e}")
        return False


def test_dataset_inspection():
    """Test dataset inspection functionality."""
    print("\n" + "=" * 60)
    print("Testing Dataset Inspection")
    print("=" * 60)

    dataset_path = "data/wave_dataset_T250.h5"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    try:
        # Test inspection
        inspect_dataset(dataset_path, num_samples=3)
        print("✓ Dataset inspection completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error testing dataset inspection: {e}")
        return False


def main():
    """Run all data pipeline tests."""
    print("Testing Wave Dataset Pipeline")
    print("=" * 60)

    tests = [
        ("Single Dataset Loading", test_single_dataset),
        ("DataLoader Creation", test_dataloader_creation),
        ("Dataset Inspection", test_dataset_inspection),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All data pipeline tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
