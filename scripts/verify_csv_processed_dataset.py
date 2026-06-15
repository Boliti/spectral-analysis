#!/usr/bin/env python3
"""Verify CSV export for processed dataset case."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analysis.run_manager import AnalysisRunManager

def test_processed_csv_export():
    """Test CSV export for processed dataset."""
    
    manager = AnalysisRunManager(root_dir=Path("user_data/test_user"))
    
    # Processed dataset run
    run = {
        "run": {
            "run_id": "run_20260615_110000",
            "created_at": "2026-06-15T11:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_def456",
            "dataset_name": "Raman_krov",
            # NO explicit dataset_version
            "used_processed_data": True,  # This time True!
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {
                "method": "normalize",
                "params": {"type": "minmax"}
            },
            "preprocessing_preset": "standard",
            "spectrometer_parameters": {"wavelength_range": "400-2000 cm-1"},
            "class_distribution": {"class_A": 50, "class_B": 50},
            "best_method": "Random Forest",
            "status": "completed",
            "warnings": [],
        },
        "results": {
            "metrics": {
                "accuracy": 0.96,
                "precision_macro": 0.96,
                "recall_macro": 0.96,
                "f1_macro": 0.96,
                "confusion_matrix": [[48, 2], [2, 48]],
                "confusion_matrix_total": 100,
                "confusion_matrix_accuracy": 0.96,
                "train_samples": 80,
                "test_samples": 100,
                "validation_mode": "hold_out",
                "classes": ["class_A", "class_B"],
            },
            "saved_model": {
                "model_type": "RandomForest",
                "model_id": "model_abc123",
            }
        }
    }
    
    row = manager._csv_row(run)
    
    print("=" * 80)
    print("PROCESSED DATASET CSV EXPORT TEST")
    print("=" * 80)
    print("\nTest scenario: Processed dataset (Raman_krov)")
    print("  - used_processed_data: True")
    print("  - dataset_version: NOT explicitly set (should be inferred)\n")
    
    print("Expected output:")
    print(f"  dataset_version: 'processed'")
    print(f"  used_processed_data: True")
    print(f"  preprocessing_config: (with actual preprocessing methods)")
    print(f"  warnings: []\n")
    
    print("Actual output:")
    print(f"  dataset_version: {repr(row['dataset_version'])}")
    print(f"  used_processed_data: {row['used_processed_data']}")
    print(f"  preprocessing_config: {row['preprocessing_config']}")
    print(f"  warnings: {row['warnings']}\n")
    
    # Assertions
    assert row['dataset_version'] == 'processed', f"FAIL: dataset_version should be 'processed', got {row['dataset_version']}"
    assert row['used_processed_data'] == True, f"FAIL: used_processed_data should be True"
    assert 'normalize' in row['preprocessing_config'], f"FAIL: preprocessing_config should contain 'normalize'"
    assert row['warnings'] == '[]', f"FAIL: warnings should be empty, got {row['warnings']}"
    
    print("✓ Key fields match expected values!")
    
    # Show additional metrics
    print("\nAdditional CSV fields:")
    print(f"  run_id: {row['run_id']}")
    print(f"  dataset_name: {row['dataset_name']}")
    print(f"  method: {row['method']}")
    print(f"  accuracy: {row['accuracy']}")
    print(f"  preprocessing_preset: {row['preprocessing_preset']}")
    
    print("\n" + "=" * 80)
    print("✓ PROCESSED DATASET CSV EXPORT TEST PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_processed_csv_export()
