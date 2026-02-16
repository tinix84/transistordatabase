import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_directory(base_path):
    """Analyze directory structure and file formats."""
    stats = {
        'total_files': 0,
        'by_format': defaultdict(int),
        'by_type': defaultdict(int),
        'by_manufacturer': defaultdict(int),
        'files': []
    }

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.json', '.xml', '.csv')):
                full_path = os.path.join(root, file)
                ext = file.split('.')[-1]

                stats['total_files'] += 1
                stats['by_format'][ext] += 1
                stats['files'].append(full_path)

                # Try to extract type and manufacturer
                if ext == 'json':
                    try:
                        with open(full_path, 'r') as f:
                            data = json.load(f)
                            if 'metadata' in data:
                                transistor_type = data['metadata'].get('type', 'unknown')
                                manufacturer = data['metadata'].get('manufacturer', 'unknown')
                            elif 'type' in data:
                                transistor_type = data.get('type', 'unknown')
                                manufacturer = data.get('manufacturer', 'unknown')
                            else:
                                transistor_type = 'unknown'
                                manufacturer = 'unknown'

                            stats['by_type'][transistor_type] += 1
                            stats['by_manufacturer'][manufacturer] += 1
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

    return stats

# Analyze all sources
print("=== Analyzing TDB ===")
tdb_stats = analyze_directory('/home/tinix/claude_wsl/transistordatabase/database')
print(json.dumps({k: v if not isinstance(v, defaultdict) else dict(v) for k, v in tdb_stats.items() if k != 'files'}, indent=2))

print("\n=== Analyzing sc archive ===")
sc_stats = analyze_directory('/home/tinix/claude_wsl/archive_ntbees2/sc')
print(json.dumps({k: v if not isinstance(v, defaultdict) else dict(v) for k, v in sc_stats.items() if k != 'files'}, indent=2))

print("\n=== Analyzing File Exchange ===")
fe_stats = analyze_directory('/home/tinix/claude_wsl/file_exchange')
print(json.dumps({k: v if not isinstance(v, defaultdict) else dict(v) for k, v in fe_stats.items() if k != 'files'}, indent=2))

# Generate final report
report = {
    'timestamp': datetime.now().isoformat(),
    'sources': {
        'tdb': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in tdb_stats.items() if k != 'files'},
        'sc': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in sc_stats.items() if k != 'files'},
        'file_exchange': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in fe_stats.items() if k != 'files'}
    },
    'totals': {
        'tdb': tdb_stats['total_files'],
        'sc': sc_stats['total_files'],
        'file_exchange': fe_stats['total_files'],
        'grand_total': tdb_stats['total_files'] + sc_stats['total_files'] + fe_stats['total_files']
    },
    'file_lists': {
        'tdb': tdb_stats['files'][:10],  # First 10 only to keep report manageable
        'sc': sc_stats['files'][:10],
        'file_exchange': fe_stats['files'][:10]
    }
}

with open('/home/tinix/claude_wsl/transistordatabase/INVENTORY_REPORT.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Inventory complete: {report['totals']['grand_total']} total transistor files found")
print(f"  - TDB: {report['totals']['tdb']}")
print(f"  - sc: {report['totals']['sc']}")
print(f"  - File Exchange: {report['totals']['file_exchange']}")
