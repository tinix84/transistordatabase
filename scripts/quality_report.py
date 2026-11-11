#!/usr/bin/env python3
"""
Generate comprehensive quality report for merged transistor database.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def generate_quality_report():
    """Generate comprehensive quality report."""

    # Load reports
    migration_path = Path('/home/tinix/claude_wsl/transistordatabase/MIGRATION_REPORT.json')
    validation_path = Path('/home/tinix/claude_wsl/transistordatabase/VALIDATION_REPORT.json')

    if not migration_path.exists():
        print(f"❌ Migration report not found: {migration_path}")
        return

    with open(migration_path, 'r') as f:
        migration = json.load(f)

    validation = None
    if validation_path.exists():
        with open(validation_path, 'r') as f:
            validation = json.load(f)

    # Analyze merged directory
    merged_dir = Path('/home/tinix/claude_wsl/transistordatabase/transistors_merged')

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_transistors': len(list(merged_dir.glob('*.json'))) if merged_dir.exists() else 0,
            'migration_success_rate': f"{migration['summary']['successful'] / migration['summary']['total_processed'] * 100:.2f}%",
            'average_quality_score': migration['summary']['average_quality_score'],
            'median_quality_score': migration['summary']['median_quality_score'],
            'conflicts': migration['summary']['duplicates_found']
        },
        'by_manufacturer': {},
        'by_type': {},
        'data_completeness': {
            'has_channel_curves': 0,
            'has_switching_losses': 0,
            'has_gate_charge': 0,
            'has_soa': 0,
            'has_capacitance': 0,
            'has_thermal_model': 0
        },
        'validation': validation['summary'] if validation else {'note': 'Validation not run'},
        'recommendations': [],
        'quality_metrics': {}
    }

    # Analyze each transistor file
    if merged_dir.exists():
        for json_file in merged_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Count by manufacturer
                mfr = data.get('metadata', {}).get('manufacturer', 'Unknown')
                report['by_manufacturer'][mfr] = report['by_manufacturer'].get(mfr, 0) + 1

                # Count by type
                transistor_type = data.get('metadata', {}).get('type', 'Unknown')
                report['by_type'][transistor_type] = report['by_type'].get(transistor_type, 0) + 1

                # Check completeness (handle both core and legacy formats)
                switch = data.get('switch', {})
                diode = data.get('diode', {})

                # Check for channel data (legacy format uses 'channel', core uses 'channel_data')
                if switch.get('channel_data') or switch.get('channel'):
                    report['data_completeness']['has_channel_curves'] += 1

                # Check for switching losses (legacy 'e_loss', core 'e_on_data'/'e_off_data')
                if switch.get('e_on_data') or switch.get('e_off_data') or switch.get('e_loss'):
                    report['data_completeness']['has_switching_losses'] += 1

                # Check for gate charge (legacy 'q_gate', core 'gate_charge_curves')
                if switch.get('gate_charge_curves') or switch.get('q_gate'):
                    report['data_completeness']['has_gate_charge'] += 1

                # Check for SOA (both formats use same structure)
                if switch.get('soa'):
                    report['data_completeness']['has_soa'] += 1

                # Check for capacitance
                if data.get('c_oss') or data.get('c_iss') or data.get('c_rss'):
                    report['data_completeness']['has_capacitance'] += 1

                # Check for thermal models
                if switch.get('thermal_foster') or diode.get('thermal_foster'):
                    report['data_completeness']['has_thermal_model'] += 1

            except Exception as e:
                print(f"Warning: Error analyzing {json_file}: {e}")

    # Generate quality metrics
    total = report['summary']['total_transistors']
    if total > 0:
        for key, value in report['data_completeness'].items():
            pct = value / total * 100
            report['quality_metrics'][key] = f"{pct:.1f}%"

    # Generate recommendations
    if total > 0:
        completeness_pct = report['data_completeness']['has_channel_curves'] / total * 100
        if completeness_pct < 50:
            report['recommendations'].append({
                'priority': 'high',
                'issue': 'Low channel curve coverage',
                'message': f"Only {completeness_pct:.1f}% of transistors have channel curves. "
                          "Consider enriching data from additional sources."
            })

        if completeness_pct > 90:
            report['recommendations'].append({
                'priority': 'info',
                'issue': 'Excellent channel curve coverage',
                'message': f"{completeness_pct:.1f}% of transistors have channel curves."
            })

        if report['summary']['conflicts'] > total * 0.1:
            report['recommendations'].append({
                'priority': 'high',
                'issue': 'High conflict rate',
                'message': f"{report['summary']['conflicts']} conflicts detected ({report['summary']['conflicts']/total*100:.1f}%). "
                          "Review CONFLICT_REVIEW.html to resolve duplicates."
            })
        else:
            report['recommendations'].append({
                'priority': 'info',
                'issue': 'Manageable conflict rate',
                'message': f"{report['summary']['conflicts']} conflicts detected ({report['summary']['conflicts']/total*100:.1f}%)."
            })

        gate_charge_pct = report['data_completeness']['has_gate_charge'] / total * 100
        if gate_charge_pct < 30:
            report['recommendations'].append({
                'priority': 'medium',
                'issue': 'Limited gate charge data',
                'message': f"Only {gate_charge_pct:.1f}% have gate charge curves. "
                          "Consider importing from PLECS or datasheet extraction."
            })

        soa_pct = report['data_completeness']['has_soa'] / total * 100
        if soa_pct < 20:
            report['recommendations'].append({
                'priority': 'medium',
                'issue': 'Limited SOA data',
                'message': f"Only {soa_pct:.1f}% have SOA curves. "
                          "Add Safe Operating Area data for simulation accuracy."
            })

    # Save report
    output_path = Path('/home/tinix/claude_wsl/transistordatabase/QUALITY_REPORT.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"QUALITY REPORT GENERATED")
    print(f"{'='*60}")
    print(f"Total Transistors: {report['summary']['total_transistors']}")
    print(f"Migration Success: {report['summary']['migration_success_rate']}")
    print(f"Average Quality: {report['summary']['average_quality_score']:.1f}%")
    print(f"Median Quality: {report['summary']['median_quality_score']:.1f}%")
    print(f"Conflicts: {report['summary']['conflicts']}")

    print(f"\nBy Type:")
    type_sorted = sorted(report['by_type'].items(), key=lambda x: x[1], reverse=True)
    for t, count in type_sorted:
        pct = count / total * 100 if total > 0 else 0
        print(f"  {t}: {count} ({pct:.1f}%)")

    print(f"\nBy Manufacturer (top 10):")
    top_mfr = sorted(report['by_manufacturer'].items(), key=lambda x: x[1], reverse=True)[:10]
    for mfr, count in top_mfr:
        pct = count / total * 100 if total > 0 else 0
        print(f"  {mfr}: {count} ({pct:.1f}%)")

    print(f"\nData Completeness:")
    for key, value in report['data_completeness'].items():
        pct = value / total * 100 if total > 0 else 0
        print(f"  {key}: {value} ({pct:.1f}%)")

    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        priority_emoji = {
            'high': 'CRITICAL',
            'medium': 'WARNING',
            'info': 'INFO'
        }.get(rec['priority'], '?')
        print(f"  [{priority_emoji}] {rec['issue']}")
        print(f"       {rec['message']}")

    print(f"\nReport saved to: {output_path}")
    print(f"{'='*60}\n")

    return report

if __name__ == "__main__":
    generate_quality_report()
