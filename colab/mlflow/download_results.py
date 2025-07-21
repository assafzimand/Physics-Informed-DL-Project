#!/usr/bin/env python3
"""
Download MLflow Results from Colab to Local Machine

This script downloads experiment results from Google Drive after Colab sessions.
Automatically syncs MLflow tracking data, trained models, and visualizations.

Usage:
    python colab/mlflow/download_results.py

Features:
    - Downloads new MLflow experiments from Drive
    - Syncs trained models 
    - Preserves experiment history
    - Organizes results by date/session
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
import zipfile


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


def create_backup(local_mlruns):
    """Create backup of existing MLflow data."""
    if not local_mlruns.exists():
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = local_mlruns.parent / f"mlruns_backup_{timestamp}"
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copytree(local_mlruns, backup_path)
    return backup_path


def download_mlflow_data():
    """Download MLflow data from Google Drive."""
    print("📥 Downloading MLflow data from Google Drive...")
    
    # Find Google Drive
    drive_path = find_google_drive_path()
    if not drive_path:
        print("❌ Google Drive not found")
        return False
    
    # Check for Colab results
    colab_results = drive_path / "Physics_Informed_DL_Project" / "results"
    if not colab_results.exists():
        print(f"⚠️  No results found at: {colab_results}")
        print("   Run some Colab experiments first!")
        return False
    
    print(f"✅ Found results directory: {colab_results}")
    
    # Local paths
    local_base = Path("colab_results")
    local_base.mkdir(exist_ok=True)
    
    # Download strategy based on what's available
    downloaded_anything = False
    
    # 1. Download MLflow tracking data
    drive_mlruns = colab_results / "mlruns"
    if drive_mlruns.exists():
        local_mlruns = local_base / "mlruns"
        
        # Backup existing data
        backup = create_backup(local_mlruns) if local_mlruns.exists() else None
        
        try:
            if local_mlruns.exists():
                shutil.rmtree(local_mlruns)
            shutil.copytree(drive_mlruns, local_mlruns)
            
            print(f"✅ Downloaded MLflow data: {local_mlruns}")
            downloaded_anything = True
            
        except Exception as e:
            print(f"❌ MLflow download error: {e}")
            if backup:
                print(f"🔄 Restoring backup...")
                if local_mlruns.exists():
                    shutil.rmtree(local_mlruns)
                shutil.move(backup, local_mlruns)
    
    # 2. Download trained models
    drive_models = colab_results / "models"
    if drive_models.exists():
        local_models = local_base / "models"
        local_models.mkdir(exist_ok=True)
        
        # Only download new models
        for model_file in drive_models.glob("*.pth"):
            local_file = local_models / model_file.name
            
            if not local_file.exists():
                shutil.copy2(model_file, local_file)
                file_size = local_file.stat().st_size / 1e6
                print(f"📦 Downloaded model: {model_file.name} ({file_size:.1f} MB)")
                downloaded_anything = True
            else:
                print(f"⏭️  Model already exists: {model_file.name}")
    
    # 3. Download plots and visualizations
    drive_plots = colab_results / "plots"
    if drive_plots.exists():
        local_plots = local_base / "plots"
        local_plots.mkdir(exist_ok=True)
        
        for plot_file in drive_plots.glob("*"):
            if plot_file.is_file():
                local_file = local_plots / plot_file.name
                
                if not local_file.exists():
                    shutil.copy2(plot_file, local_file)
                    print(f"📊 Downloaded plot: {plot_file.name}")
                    downloaded_anything = True
    
    return downloaded_anything


def organize_results():
    """Organize downloaded results by date."""
    results_base = Path("colab_results")
    if not results_base.exists():
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = results_base / f"session_{timestamp}"
    
    # Move current downloads to session directory
    if any(results_base.iterdir()):
        session_dir.mkdir(exist_ok=True)
        
        for item in results_base.iterdir():
            if item.name.startswith("session_"):
                continue
            
            target = session_dir / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            
            shutil.move(item, target)
        
        print(f"📁 Organized results in: {session_dir}")


def summarize_download():
    """Summarize what was downloaded."""
    results_base = Path("colab_results")
    if not results_base.exists():
        return
    
    print("\n📊 Download Summary:")
    print("=" * 40)
    
    # Count MLflow experiments
    mlruns = results_base / "mlruns"
    if mlruns.exists():
        experiments = len([d for d in mlruns.iterdir() if d.is_dir() and d.name.isdigit()])
        print(f"🧪 MLflow experiments: {experiments}")
    
    # Count models
    models = results_base / "models"
    if models.exists():
        model_files = list(models.glob("*.pth"))
        total_size = sum(f.stat().st_size for f in model_files) / 1e6
        print(f"🤖 Trained models: {len(model_files)} ({total_size:.1f} MB)")
    
    # Count plots
    plots = results_base / "plots"
    if plots.exists():
        plot_files = len([f for f in plots.iterdir() if f.is_file()])
        print(f"📈 Visualizations: {plot_files}")
    
    print("\n🎯 Results saved to: colab_results/")


def main():
    """Main download function."""
    print("📥 MLflow Results Download from Google Colab")
    print("=" * 50)
    
    success = download_mlflow_data()
    
    if success:
        organize_results()
        summarize_download()
        
        print("\n✅ Download completed successfully!")
        print("\n📖 What you can do now:")
        print("   1. Analyze results: python scripts/analyze_experiments.py")
        print("   2. Use best model: Load from colab_results/models/")
        print("   3. View MLflow UI: mlflow ui --backend-store-uri colab_results/mlruns")
        
    else:
        print("\n⚠️  No new results to download")
        print("   Make sure you ran experiments in Colab first")


if __name__ == "__main__":
    main() 