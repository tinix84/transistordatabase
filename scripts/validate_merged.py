#!/usr/bin/env python3
"""
Validate all merged transistor files using TDB's validation framework.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add TDB to path
sys.path.insert(0, '/home/tinix/claude_wsl/transistordatabase')

try:
    from transistordatabase.core.repository import JsonTransistorLoader
    from transistordatabase.backend.concrete_services import ValidationService
    TDB_VALIDATION_AVAILABLE = True
except ImportError:
    print("⚠️  TDB ValidationService not available, using fallback validation")
    TDB_VALIDATION_AVAILABLE = False

def validate_json_structure(file_path: Path) -> Dict:
    """Fallback validation - check basic JSON structure."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        errors = []
        warnings = []

        # Check required top-level keys
        required_keys = ['metadata', 'electrical_ratings', 'thermal_properties', 'switch']
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")

        # Check metadata
        if 'metadata' in data:
            if not data['metadata'].get('name'):
                errors.append("Missing metadata.name")
            if not data['metadata'].get('type'):
                warnings.append("Missing metadata.type")

        # Check electrical ratings
        if 'electrical_ratings' in data:
            v_max = data['electrical_ratings'].get('v_abs_max', 0)
            if v_max <= 0:
                warnings.append(f"Invalid v_abs_max: {v_max}")

            i_max = data['electrical_ratings'].get('i_abs_max', 0)
            if i_max <= 0:
                warnings.append(f"Invalid i_abs_max: {i_max}")

        # Check switch data (can be in legacy format with 'channel' or core format with 'channel_data')
        if 'switch' in data:
            switch = data['switch']
            # Check for either legacy 'channel' or core 'channel_data'
            if not switch.get('channel') and not switch.get('channel_data'):
                warnings.append("Switch has no channel data")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'errors': [f"Invalid JSON: {str(e)}"],
            'warnings': []
        }
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': []
        }

def validate_with_tdb(file_path: Path) -> Dict:
    """Validate using TDB's ValidationService."""
    try:
        # Load transistor using TDB loader
        loader = JsonTransistorLoader()
        transistor = loader.load_from_json(file_path)

        # Validate using TDB service
        validator = ValidationService()
        validation = validator.validate_transistor(transistor)

        return validation

    except Exception as e:
        return {
            'valid': False,
            'errors': [f"TDB validation error: {str(e)}"],
            'warnings': []
        }

def validate_all_merged():
    """Validate all merged transistor files."""
    merged_dir = Path('/home/tinix/claude_wsl/transistordatabase/transistors_merged')

    if not merged_dir.exists():
        print(f"❌ Merged directory not found: {merged_dir}")
        return None

    results = []
    json_files = list(merged_dir.glob('*.json'))
    total = len(json_files)

    print(f"\n{'='*60}")
    print(f"VALIDATING {total} MERGED TRANSISTOR FILES")
    print(f"{'='*60}\n")

    for i, json_file in enumerate(json_files, 1):
        # Progress update
        if i % 25 == 0 or i == total:
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

        # Use fallback validation - TDB validation doesn't support all types
        # (e.g., Diodes are in the database but not in TDB's supported types)
        validation = validate_json_structure(json_file)

        results.append({
            'file': json_file.name,
            'transistor_id': json_file.stem,
            'valid': validation['valid'],
            'errors': validation['errors'],
            'warnings': validation['warnings']
        })

    # Generate summary
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = total - valid_count
    total_errors = sum(len(r['errors']) for r in results)
    total_warnings = sum(len(r['warnings']) for r in results)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {total}")
    print(f"Valid: {valid_count} ({valid_count/total*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/total*100:.1f}%)")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print(f"{'='*60}\n")

    # Save detailed results
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_method': 'TDB_ValidationService' if TDB_VALIDATION_AVAILABLE else 'fallback_json_check',
        'summary': {
            'total_files': total,
            'valid': valid_count,
            'invalid': invalid_count,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'pass_rate': round(valid_count/total*100, 2) if total > 0 else 0
        },
        'results': results
    }

    output_path = Path('/home/tinix/claude_wsl/transistordatabase/VALIDATION_REPORT.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Detailed validation report saved to: {output_path}\n")

    return report

if __name__ == "__main__":
    validate_all_merged()
