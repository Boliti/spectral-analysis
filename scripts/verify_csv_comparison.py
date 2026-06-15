#!/usr/bin/env python3
"""Verify CSV export for comparison run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analysis.run_manager import AnalysisRunManager

def test_comparison_csv_export():
    """Test CSV export for comparison run with multiple methods."""
    
    manager = AnalysisRunManager(root_dir=Path("user_data/test_user"))
    
    # Comparison run with multiple methods
    run = {
        "run": {
            "run_id": "run_comparison_001",
            "created_at": "2026-06-15T12:00:00",
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
            "status": "completed",
            "warnings": [],
        },
        "run_type": "comparison",
        "created_at": "2026-06-15T12:00:00",
        "results": {
            "comparison": [
                {
                    "method": "SVM",
                    "model_type": "SVM",
                    "model_id": "model_svm_001",
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
                    "status": "completed",
                    "warnings": [],
                },
                {
                    "method": "Random Forest",
                    "model_type": "RandomForest",
                    "model_id": "model_rf_001",
                    "metrics": {
                        "accuracy": 0.94,
                        "precision_macro": 0.94,
                        "recall_macro": 0.94,
                        "f1_macro": 0.94,
                        "confusion_matrix": [[47, 3], [3, 47]],
                        "confusion_matrix_total": 100,
                        "confusion_matrix_accuracy": 0.94,
                        "train_samples": 80,
                        "test_samples": 100,
                        "validation_mode": "hold_out",
                        "classes": ["class_A", "class_B"],
                    },
                    "status": "completed",
                    "warnings": [],
                }
            ]
        }
    }
    
    rows = manager._csv_rows_for_run(run)
    
    print("=" * 80)
    print("COMPARISON RUN CSV EXPORT TEST")
    print("=" * 80)
    print("\nTest scenario: Comparison run (raw dataset, 2 methods)")
    print("  - run_type: 'comparison'")
    print("  - Methods: SVM, Random Forest")
    print("  - used_processed_data: False\n")
    
    assert len(rows) == 2, f"FAIL: Expected 2 rows, got {len(rows)}"
    print(f"Expected: 2 CSV rows (one per method)")
    print(f"Actual: {len(rows)} rows\n")
    
    for i, row in enumerate(rows, 1):
        print(f"Row {i} ({row['method']}):")
        print(f"  dataset_version: {repr(row['dataset_version'])}")
        print(f"  used_processed_data: {row['used_processed_data']}")
        print(f"  accuracy: {row['accuracy']}")
        print(f"  warnings: {row['warnings']}")
        
        # Assertions for each method row
        assert row['dataset_version'] == 'raw', f"FAIL: dataset_version should be 'raw', got {row['dataset_version']}"
        assert row['used_processed_data'] == False, f"FAIL: used_processed_data should be False"
        assert row['warnings'] == '[]', f"FAIL: warnings should be empty, got {row['warnings']}"
        print(f"  ✓ Fields correct\n")
    
    print("=" * 80)
    print("✓ COMPARISON RUN CSV EXPORT TEST PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_comparison_csv_export()
