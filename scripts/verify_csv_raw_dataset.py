#!/usr/bin/env python3
"""Verify complete CSV row output with dataset_version inference."""

import sys
import json
import csv
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analysis.run_manager import AnalysisRunManager

def test_complete_csv_export():
    """Test complete CSV export with all fields for raw dataset case."""
    
    manager = AnalysisRunManager(root_dir=Path("user_data/test_user"))
    
    # Realistic raw dataset run (matching user's scenario)
    run = {
        "run": {
            "run_id": "run_20260615_100000",
            "created_at": "2026-06-15T10:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_abc123",
            "dataset_name": "Raman_krov_raw",
            # NO explicit dataset_version
            "used_processed_data": False,
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {},
            "preprocessing_preset": "",
            "spectrometer_parameters": {"wavelength_range": "400-2000 cm-1"},
            "class_distribution": {"class_A": 50, "class_B": 50},
            "best_method": "SVM",
            "status": "completed",
            "warnings": [],  # Initial empty warnings
        },
        "results": {
            "metrics": {
                "accuracy": 0.91,
                "precision_macro": 0.91,
                "recall_macro": 0.91,
                "f1_macro": 0.91,
                "confusion_matrix": [[46, 4], [5, 45]],
                "confusion_matrix_total": 100,
                "confusion_matrix_accuracy": 0.91,
                "train_samples": 80,
                "test_samples": 100,
                "validation_mode": "hold_out",
                "classes": ["class_A", "class_B"],
            },
            "saved_model": {
                "model_type": "SVM",
                "model_id": "model_xyz789",
            }
        }
    }
    
    row = manager._csv_row(run)
    
    print("=" * 80)
    print("RAW DATASET CSV EXPORT TEST")
    print("=" * 80)
    print("\nTest scenario: Raw dataset (Raman_krov_raw)")
    print("  - used_processed_data: False")
    print("  - dataset_version: NOT explicitly set (should be inferred)\n")
    
    print("Expected output:")
    print(f"  dataset_version: 'raw'")
    print(f"  used_processed_data: False")
    print(f"  preprocessing_config: {{}}")
    print(f"  warnings: []\n")
    
    print("Actual output:")
    print(f"  dataset_version: {repr(row['dataset_version'])}")
    print(f"  used_processed_data: {row['used_processed_data']}")
    print(f"  preprocessing_config: {row['preprocessing_config']}")
    print(f"  warnings: {row['warnings']}\n")
    
    # Assertions
    assert row['dataset_version'] == 'raw', f"FAIL: dataset_version should be 'raw', got {row['dataset_version']}"
    assert row['used_processed_data'] == False, f"FAIL: used_processed_data should be False"
    assert row['preprocessing_config'] == '{}', f"FAIL: preprocessing_config should be '{{}}'"
    assert row['warnings'] == '[]', f"FAIL: warnings should be empty, got {row['warnings']}"
    
    print("✓ Key fields match expected values!")
    
    # Show additional metrics to verify CSV completeness
    print("\nAdditional CSV fields:")
    print(f"  run_id: {row['run_id']}")
    print(f"  created_at: {row['created_at']}")
    print(f"  run_type: {row['run_type']}")
    print(f"  dataset_name: {row['dataset_name']}")
    print(f"  target_name: {row['target_name']}")
    print(f"  method: {row['method']}")
    print(f"  model_type: {row['model_type']}")
    print(f"  accuracy: {row['accuracy']}")
    print(f"  confusion_matrix_total: {row['confusion_matrix_total']}")
    print(f"  test_samples: {row['test_samples']}")
    print(f"  status: {row['status']}")
    
    print("\n" + "=" * 80)
    print("✓ CSV EXPORT TEST PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_complete_csv_export()
