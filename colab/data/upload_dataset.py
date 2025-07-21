#!/usr/bin/env python3
"""
Upload Dataset to Google Drive for Colab Experiments

This script uploads the local wave dataset to Google Drive so it can be 
accessed from Google Colab experiments.

Usage:
    python colab/data/upload_dataset.py

Requirements:
    - Dataset file: data/wave_dataset_T500.h5
    - Google Drive mounted/synced on local machine
"""

import os
import shutil
import sys
from pathlib import Path


def find_google_drive_path():
    """Find Google Drive path on local machine."""
    possible_paths = [
        # Windows
        Path.home() / "Google Drive" / "My Drive",
        Path("G:") / "My Drive",
        Path("H:") / "My Drive",
        # macOS
        Path.home() / "Google Drive",
        # Linux
        Path.home() / "GoogleDrive",
        Path.home() / "Google Drive",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def main():
    """Main upload function."""
    print("🌊 Wave Dataset Upload to Google Drive")
    print("=" * 50)
    
    # 1. Check local dataset exists
    local_dataset = Path("data/wave_dataset_T500.h5")
    if not local_dataset.exists():
        print(f"❌ Dataset not found at: {local_dataset}")
        print("   Make sure you're running from the project root directory")
        print("   Expected structure: project_root/data/wave_dataset_T500.h5")
        return False
    
    file_size_mb = local_dataset.stat().st_size / 1e6
    print(f"📊 Found dataset: {local_dataset} ({file_size_mb:.1f} MB)")
    
    # 2. Find Google Drive
    drive_path = find_google_drive_path()
    if not drive_path:
        print("❌ Google Drive not found on this system")
        print("   Please install and sync Google Drive first")
        print("   Or manually copy the dataset to:")
        print("   Google Drive/Physics_Informed_DL_Project/datasets/")
        return False
    
    print(f"✅ Found Google Drive at: {drive_path}")
    
    # 3. Create target directory
    target_dir = drive_path / "Physics_Informed_DL_Project" / "datasets"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created directory: {target_dir}")
    
    # 4. Upload dataset
    target_file = target_dir / "wave_dataset_T500.h5"
    
    if target_file.exists():
        print(f"⚠️  Dataset already exists in Drive")
        response = input("   Overwrite? (y/N): ").lower().strip()
        if response != 'y':
            print("   Upload cancelled")
            return True
        
        # Remove existing file
        target_file.unlink()
        print("   Removed existing file")
    
    print(f"📤 Uploading dataset to Google Drive...")
    print(f"   From: {local_dataset}")
    print(f"   To:   {target_file}")
    
    try:
        # Copy file
        shutil.copy2(local_dataset, target_file)
        
        # Verify upload
        if target_file.exists():
            uploaded_size_mb = target_file.stat().st_size / 1e6
            print(f"✅ Upload successful! ({uploaded_size_mb:.1f} MB)")
            
            # Verify file integrity
            if abs(file_size_mb - uploaded_size_mb) < 0.1:
                print("✅ File integrity verified")
            else:
                print(f"⚠️  Size mismatch: {file_size_mb:.1f} MB vs {uploaded_size_mb:.1f} MB")
            
            return True
        else:
            print("❌ Upload failed - file not found after copy")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False


def verify_upload():
    """Verify the uploaded dataset."""
    print("\n🔍 Verifying uploaded dataset...")
    
    drive_path = find_google_drive_path()
    if not drive_path:
        return False
    
    target_file = drive_path / "Physics_Informed_DL_Project" / "datasets" / "wave_dataset_T500.h5"
    
    if not target_file.exists():
        print("❌ Dataset not found in Drive")
        return False
    
    try:
        import h5py
        with h5py.File(target_file, 'r') as f:
            keys = list(f.keys())
            wave_shape = f['wave_fields'].shape
            coord_shape = f['source_coords'].shape
            
        print(f"✅ Dataset verification successful:")
        print(f"   📊 Keys: {keys}")
        print(f"   📏 Wave fields: {wave_shape}")
        print(f"   📍 Coordinates: {coord_shape}")
        return True
        
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting dataset upload process...")
    
    success = main()
    
    if success:
        verify_upload()
        print("\n🎉 Upload complete! You can now run Colab experiments.")
        print("\n📖 Next steps:")
        print("   1. Open Google Colab")
        print("   2. Upload colab/setup/colab_setup.ipynb")
        print("   3. Run the setup notebook")
        print("   4. Start training!")
    else:
        print("\n❌ Upload failed. Please check the errors above.")
        sys.exit(1) 