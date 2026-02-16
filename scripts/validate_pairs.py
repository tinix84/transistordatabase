#!/usr/bin/env python3
"""Validate all switching pair JSON files and generate comprehensive reports."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/tinix/claude_wsl/transistordatabase')

from transistordatabase.core.pair_models import SwitchingPair


def validate_all_pairs(pairs_dir: Path) -> dict:
    """Validate all switching pair JSON files."""
    pairs_dir = Path(pairs_dir)
    json_files = sorted(pairs_dir.glob('*.json'))
    total = len(json_files)

    print(f"\n{'='*60}")
    print(f"VALIDATING {total} SWITCHING PAIR FILES")
    print(f"{'='*60}\n")

    results = []
    stats = {
        'total': total,
        'valid': 0,
        'invalid': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'by_pair_type': defaultdict(int),
        'by_source': defaultdict(int),
        'by_manufacturer': defaultdict(int),
        'by_device_type': defaultdict(int),
        'quality_distribution': {
            'excellent_90_100': 0,
            'good_70_89': 0,
            'fair_50_69': 0,
            'poor_below_50': 0,
        },
        'data_completeness': {
            'has_conduction_data': 0,
            'has_switching_energy': 0,
            'has_thermal_model': 0,
            'has_plecs_formula': 0,
            'has_gate_charge': 0,
            'has_capacitance': 0,
            'has_soa': 0,
        },
        'self_pairs': 0,
        'combo_pairs': 0,
        'parallel_pairs': 0,
        'unique_devices': set(),
        'quality_scores': [],
    }

    for i, json_file in enumerate(json_files, 1):
        if i % 25 == 0 or i == total:
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            pair = SwitchingPair.from_dict(data)
            validation = pair.validate()

            # Collect validation results
            entry = {
                'file': json_file.name,
                'pair_id': pair.pair_id,
                'pair_type': pair.pair_type,
                'valid': validation['valid'],
                'errors': validation['errors'],
                'warnings': validation['warnings'],
                'quality_score': validation['quality_score'],
                'is_self_pair': pair.is_self_pair,
                'is_parallel': pair.is_parallel,
                'devices': pair.devices,
            }
            results.append(entry)

            # Update statistics
            if validation['valid']:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1

            stats['total_errors'] += len(validation['errors'])
            stats['total_warnings'] += len(validation['warnings'])
            stats['by_pair_type'][pair.pair_type] += 1
            stats['quality_scores'].append(validation['quality_score'])

            # Source tracking
            source = data.get('switching_data', {}).get('source', 'unknown')
            stats['by_source'][source] += 1

            # Device tracking
            hs = data.get('high_side', {})
            ls = data.get('low_side', {})
            hs_meta = hs.get('metadata', {})

            manufacturer = hs_meta.get('manufacturer', 'Unknown')
            stats['by_manufacturer'][manufacturer] += 1

            device_type = hs_meta.get('type', 'Unknown')
            stats['by_device_type'][device_type] += 1

            # Unique devices
            if hs.get('device_id'):
                stats['unique_devices'].add(hs['device_id'])
            if ls.get('device_id'):
                stats['unique_devices'].add(ls['device_id'])

            # Self/combo/parallel classification
            if pair.is_self_pair:
                stats['self_pairs'] += 1
            elif pair.is_parallel:
                stats['parallel_pairs'] += 1
            else:
                stats['combo_pairs'] += 1

            # Quality distribution
            score = validation['quality_score']
            if score >= 90:
                stats['quality_distribution']['excellent_90_100'] += 1
            elif score >= 70:
                stats['quality_distribution']['good_70_89'] += 1
            elif score >= 50:
                stats['quality_distribution']['fair_50_69'] += 1
            else:
                stats['quality_distribution']['poor_below_50'] += 1

            # Data completeness
            if hs.get('conduction', {}).get('curves'):
                stats['data_completeness']['has_conduction_data'] += 1

            sw = data.get('switching_data', {})
            if sw.get('turn_on', {}).get('current_axis') or sw.get('turn_off', {}).get('current_axis'):
                stats['data_completeness']['has_switching_energy'] += 1

            therm = hs.get('thermal_properties', {})
            if therm.get('thermal_model') and therm['thermal_model'].get('elements'):
                stats['data_completeness']['has_thermal_model'] += 1

            plecs = data.get('plecs_model', {})
            if plecs.get('turn_on_formula') or plecs.get('conduction_formula'):
                stats['data_completeness']['has_plecs_formula'] += 1

            if hs.get('gate_charge'):
                stats['data_completeness']['has_gate_charge'] += 1

            cap = hs.get('capacitance', {})
            if cap.get('c_oss') or cap.get('c_iss') or cap.get('c_rss'):
                stats['data_completeness']['has_capacitance'] += 1

            if hs.get('soa'):
                stats['data_completeness']['has_soa'] += 1

        except Exception as e:
            results.append({
                'file': json_file.name,
                'pair_id': json_file.stem,
                'valid': False,
                'errors': [f"Failed to load: {str(e)}"],
                'warnings': [],
                'quality_score': 0,
            })
            stats['invalid'] += 1

    # Calculate averages
    avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0

    # Convert sets for JSON serialization
    stats['unique_devices'] = sorted(list(stats['unique_devices']))
    stats['unique_device_count'] = len(stats['unique_devices'])
    stats['average_quality_score'] = round(avg_quality, 2)

    return {'stats': stats, 'results': results}


def generate_reports(validation_data: dict, output_dir: Path):
    """Generate all reports from validation data."""
    stats = validation_data['stats']
    results = validation_data['results']
    total = stats['total']

    # 1. Detailed validation report (JSON)
    report_path = output_dir / 'PAIR_VALIDATION_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_pairs': total,
                'valid': stats['valid'],
                'invalid': stats['invalid'],
                'pass_rate': f"{stats['valid']/total*100:.1f}%" if total else "0%",
                'average_quality': stats['average_quality_score'],
                'unique_devices': stats['unique_device_count'],
            },
            'results': results,
        }, f, indent=2)
    print(f"  Saved: {report_path}")

    # 2. Quality report (JSON)
    quality_path = output_dir / 'PAIR_QUALITY_REPORT.json'
    with open(quality_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_pairs': total,
                'average_quality': stats['average_quality_score'],
                'self_pairs': stats['self_pairs'],
                'combo_pairs': stats['combo_pairs'],
                'parallel_pairs': stats['parallel_pairs'],
                'unique_devices': stats['unique_device_count'],
            },
            'by_pair_type': dict(stats['by_pair_type']),
            'by_source': dict(stats['by_source']),
            'by_manufacturer': dict(stats['by_manufacturer']),
            'by_device_type': dict(stats['by_device_type']),
            'quality_distribution': stats['quality_distribution'],
            'data_completeness': stats['data_completeness'],
        }, f, indent=2)
    print(f"  Saved: {quality_path}")

    # 3. Human-readable summary (Markdown)
    summary_path = output_dir / 'PAIR_DATABASE_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write("# Switching-Pair Database Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total switching pairs | {total} |\n")
        f.write(f"| Valid | {stats['valid']} ({stats['valid']/total*100:.1f}%) |\n") if total else None
        f.write(f"| Invalid | {stats['invalid']} |\n")
        f.write(f"| Average quality score | {stats['average_quality_score']:.1f}% |\n")
        f.write(f"| Unique devices | {stats['unique_device_count']} |\n")
        f.write(f"| Self-pairs | {stats['self_pairs']} |\n")
        f.write(f"| Combination pairs | {stats['combo_pairs']} |\n")
        f.write(f"| Parallel pairs | {stats['parallel_pairs']} |\n\n")

        f.write("## Pair Type Distribution\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for ptype, count in sorted(stats['by_pair_type'].items(), key=lambda x: -x[1]):
            f.write(f"| {ptype} | {count} |\n")

        f.write("\n## Data Source Distribution\n\n")
        f.write("| Source | Count |\n")
        f.write("|--------|-------|\n")
        for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
            f.write(f"| {source} | {count} |\n")

        f.write("\n## Manufacturer Distribution\n\n")
        f.write("| Manufacturer | Count |\n")
        f.write("|-------------|-------|\n")
        for mfr, count in sorted(stats['by_manufacturer'].items(), key=lambda x: -x[1]):
            f.write(f"| {mfr} | {count} |\n")

        f.write("\n## Quality Score Distribution\n\n")
        f.write("| Range | Count |\n")
        f.write("|-------|-------|\n")
        for label, count in stats['quality_distribution'].items():
            f.write(f"| {label.replace('_', ' ')} | {count} |\n")

        f.write("\n## Data Completeness\n\n")
        f.write("| Data Type | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for key, count in stats['data_completeness'].items():
            pct = count / total * 100 if total else 0
            f.write(f"| {key.replace('_', ' ').replace('has ', '')} | {count} | {pct:.1f}% |\n")

        if stats['invalid'] > 0:
            f.write("\n## Invalid Pairs\n\n")
            for r in results:
                if not r.get('valid', True):
                    f.write(f"- **{r['pair_id']}**: {', '.join(r.get('errors', []))}\n")

    print(f"  Saved: {summary_path}")

    # 4. Skipped files report (if any invalid)
    invalid_entries = [r for r in results if not r.get('valid', True)]
    if invalid_entries:
        skipped_path = output_dir / 'PAIR_ISSUES.json'
        with open(skipped_path, 'w') as f:
            json.dump(invalid_entries, f, indent=2)
        print(f"  Saved: {skipped_path}")


def main() -> None:
    """Run validation and generate reports for all switching pairs."""
    pairs_dir = Path('/home/tinix/claude_wsl/transistordatabase/switching_pairs')
    output_dir = Path('/home/tinix/claude_wsl/transistordatabase')

    if not pairs_dir.exists():
        print(f"ERROR: {pairs_dir} does not exist")
        sys.exit(1)

    # Validate
    validation_data = validate_all_pairs(pairs_dir)
    stats = validation_data['stats']

    # Print summary
    total = stats['total']
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs: {total}")
    print(f"Valid: {stats['valid']} ({stats['valid']/total*100:.1f}%)" if total else "Valid: 0")
    print(f"Invalid: {stats['invalid']}")
    print(f"Average quality: {stats['average_quality_score']:.1f}%")
    print(f"Unique devices: {stats['unique_device_count']}")
    print(f"\nBy pair type: {dict(stats['by_pair_type'])}")
    print(f"By source: {dict(stats['by_source'])}")
    print("\nData completeness:")
    for key, count in stats['data_completeness'].items():
        pct = count / total * 100 if total else 0
        print(f"  {key}: {count} ({pct:.1f}%)")
    print(f"{'='*60}\n")

    # Generate reports
    print("Generating reports...")
    generate_reports(validation_data, output_dir)

    print(f"\n{'='*60}")
    print("PHASE 1 MIGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total switching pairs: {total}")
    print(f"Validation pass rate: {stats['valid']/total*100:.1f}%" if total else "N/A")
    print(f"Average quality: {stats['average_quality_score']:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
