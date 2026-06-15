#!/usr/bin/env python3
"""Test CSV export with dataset_version inference."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.analysis.run_manager import AnalysisRunManager

def test_dataset_version_inference():
    """Test that dataset_version is correctly inferred from used_processed_data."""
    
    manager = AnalysisRunManager(root_dir=Path("user_data/test_user"))
    
    # Test case 1: Explicit dataset_version in run
    run1 = {
        "run": {
            "run_id": "test_001",
            "created_at": "2026-06-15T10:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_001",
            "dataset_name": "Raman_krov_raw",
            "dataset_version": "raw",  # Explicit
            "used_processed_data": False,
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {},
            "method": "SVM",
        },
        "results": {
            "metrics": {
                "accuracy": 0.95,
                "precision_macro": 0.94,
                "recall_macro": 0.93,
                "f1_macro": 0.93,
                "test_samples": 100,
            },
            "saved_model": {
                "model_type": "SVM",
                "model_id": "model_001",
            }
        }
    }
    
    row1 = manager._csv_row(run1)
    print("Test 1 - Explicit dataset_version:")
    print(f"  dataset_version: {row1['dataset_version']}")
    print(f"  warnings: {row1['warnings']}")
    assert row1['dataset_version'] == 'raw', f"Expected 'raw', got {row1['dataset_version']}"
    print("  ✓ PASS\n")
    
    # Test case 2: Infer from used_processed_data = False
    run2 = {
        "run": {
            "run_id": "test_002",
            "created_at": "2026-06-15T10:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_001",
            "dataset_name": "Raman_krov_raw",
            # No dataset_version key
            "used_processed_data": False,
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {},
            "method": "SVM",
        },
        "results": {
            "metrics": {
                "accuracy": 0.95,
                "precision_macro": 0.94,
                "recall_macro": 0.93,
                "f1_macro": 0.93,
                "test_samples": 100,
            },
            "saved_model": {
                "model_type": "SVM",
                "model_id": "model_002",
            }
        }
    }
    
    row2 = manager._csv_row(run2)
    print("Test 2 - Infer from used_processed_data=False:")
    print(f"  dataset_version: {row2['dataset_version']}")
    print(f"  warnings: {row2['warnings']}")
    assert row2['dataset_version'] == 'raw', f"Expected 'raw', got {row2['dataset_version']}"
    assert "dataset_version отсутствует" not in str(row2['warnings']), f"Unexpected warning: {row2['warnings']}"
    print("  ✓ PASS\n")
    
    # Test case 3: Infer from used_processed_data = True
    run3 = {
        "run": {
            "run_id": "test_003",
            "created_at": "2026-06-15T10:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_001",
            "dataset_name": "Raman_krov",
            # No dataset_version key
            "used_processed_data": True,
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {"method": "normalize"},
            "method": "SVM",
        },
        "results": {
            "metrics": {
                "accuracy": 0.97,
                "precision_macro": 0.96,
                "recall_macro": 0.96,
                "f1_macro": 0.96,
                "test_samples": 100,
            },
            "saved_model": {
                "model_type": "SVM",
                "model_id": "model_003",
            }
        }
    }
    
    row3 = manager._csv_row(run3)
    print("Test 3 - Infer from used_processed_data=True:")
    print(f"  dataset_version: {row3['dataset_version']}")
    print(f"  warnings: {row3['warnings']}")
    assert row3['dataset_version'] == 'processed', f"Expected 'processed', got {row3['dataset_version']}"
    assert "dataset_version отсутствует" not in str(row3['warnings']), f"Unexpected warning: {row3['warnings']}"
    print("  ✓ PASS\n")
    
    # Test case 4: Infer from summary.version when used_processed_data is missing
    run4 = {
        "run": {
            "run_id": "test_004",
            "created_at": "2026-06-15T10:00:00",
            "run_type": "single_model",
            "dataset_id": "dataset_001",
            "dataset_name": "Raman_krov",
            # No dataset_version and no used_processed_data
            "target_name": "class",
            "target_type": "classification",
            "preprocessing_config": {},
            "method": "SVM",
            "summary": {
                "version": "processed"
            }
        },
        "results": {
            "metrics": {
                "accuracy": 0.97,
                "precision_macro": 0.96,
                "recall_macro": 0.96,
                "f1_macro": 0.96,
                "test_samples": 100,
            },
            "saved_model": {
                "model_type": "SVM",
                "model_id": "model_004",
            }
        }
    }
    
    row4 = manager._csv_row(run4)
    print("Test 4 - Infer from summary.version:")
    print(f"  dataset_version: {row4['dataset_version']}")
    print(f"  warnings: {row4['warnings']}")
    assert row4['dataset_version'] == 'processed', f"Expected 'processed', got {row4['dataset_version']}"
    print("  ✓ PASS\n")
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_dataset_version_inference()
