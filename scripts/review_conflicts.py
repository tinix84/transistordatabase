#!/usr/bin/env python3
"""
Generate interactive HTML interface for reviewing merge conflicts.
"""

import json
from pathlib import Path
from datetime import datetime

def load_conflict_data():
    """Load conflict data from CONFLICTS_FOR_REVIEW.json"""
    conflicts_path = Path('/home/tinix/claude_wsl/transistordatabase/CONFLICTS_FOR_REVIEW.json')

    if not conflicts_path.exists():
        print(f"❌ Conflicts file not found: {conflicts_path}")
        return []

    with open(conflicts_path, 'r') as f:
        conflicts = json.load(f)

    return conflicts

def load_transistor_preview(file_path: str) -> str:
    """Load transistor data for preview."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Generate preview summary
        metadata = data.get('metadata', {})
        electrical = data.get('electrical_ratings', {})
        switch = data.get('switch', {})

        preview = f"""
        <strong>Name:</strong> {metadata.get('name', 'N/A')}<br>
        <strong>Type:</strong> {metadata.get('type', 'N/A')}<br>
        <strong>Manufacturer:</strong> {metadata.get('manufacturer', 'N/A')}<br>
        <strong>V_max:</strong> {electrical.get('v_abs_max', 0):.1f} V<br>
        <strong>I_max:</strong> {electrical.get('i_abs_max', 0):.1f} A<br>
        <strong>Channel curves:</strong> {len(switch.get('channel_data', []))}<br>
        <strong>E_on curves:</strong> {len(switch.get('e_on_data', []))}<br>
        <strong>E_off curves:</strong> {len(switch.get('e_off_data', []))}<br>
        <strong>Gate charge:</strong> {len(switch.get('gate_charge_curves', []))}<br>
        <strong>Import source:</strong> {metadata.get('import_source', 'N/A')}<br>
        """

        return preview

    except Exception as e:
        return f"<em>Error loading preview: {e}</em>"

def generate_conflict_review_html():
    """Generate HTML page for reviewing conflicts."""
    conflicts = load_conflict_data()

    if not conflicts:
        print("⚠️  No conflicts to review")
        return

    # Sort conflicts by score difference (highest first)
    conflicts.sort(key=lambda c: abs(c.get('score_difference', 0)), reverse=True)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transistor Database Merge Conflicts - Review Interface</title>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 30px;
            }}

            h1 {{
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}

            .summary {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                border-left: 5px solid #667eea;
            }}

            .summary-stats {{
                display: flex;
                gap: 30px;
                flex-wrap: wrap;
            }}

            .stat {{
                flex: 1;
                min-width: 150px;
            }}

            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}

            .stat-label {{
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }}

            .conflict {{
                border: 2px solid #e0e0e0;
                padding: 25px;
                margin: 25px 0;
                border-radius: 8px;
                background: white;
                transition: all 0.3s;
            }}

            .conflict:hover {{
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                border-color: #667eea;
            }}

            .conflict-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }}

            .transistor-id {{
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }}

            .score-badge {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: bold;
            }}

            .comparison {{
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }}

            .version {{
                flex: 1;
                padding: 20px;
                border-radius: 8px;
                border: 2px solid transparent;
                transition: all 0.3s;
            }}

            .version:hover {{
                transform: translateY(-2px);
            }}

            .existing {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
                border-color: #fdcb6e;
            }}

            .new {{
                background: linear-gradient(135deg, #a8e6cf 0%, #56ccf2 100%);
                border-color: #56ccf2;
            }}

            .version h3 {{
                margin-bottom: 15px;
                color: #333;
                font-size: 1.3em;
            }}

            .file-path {{
                font-family: 'Courier New', monospace;
                background: rgba(255,255,255,0.5);
                padding: 10px;
                border-radius: 5px;
                font-size: 0.85em;
                word-break: break-all;
                margin: 10px 0;
            }}

            .recommendation {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #4caf50;
                margin: 20px 0;
            }}

            .recommendation strong {{
                color: #2e7d32;
            }}

            .actions {{
                display: flex;
                gap: 15px;
                margin-top: 20px;
                padding-top: 20px;
                border-top: 2px solid #f0f0f0;
                flex-wrap: wrap;
            }}

            button {{
                flex: 1;
                min-width: 150px;
                padding: 12px 25px;
                border: none;
                border-radius: 6px;
                font-size: 1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}

            .btn-keep-existing {{
                background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
                color: white;
            }}

            .btn-keep-new {{
                background: linear-gradient(135deg, #56ccf2 0%, #2d98da 100%);
                color: white;
            }}

            .btn-merge {{
                background: linear-gradient(135deg, #a8e6cf 0%, #4caf50 100%);
                color: white;
            }}

            .btn-skip {{
                background: #e0e0e0;
                color: #666;
            }}

            .decision-log {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
                max-width: 300px;
                max-height: 400px;
                overflow-y: auto;
                z-index: 1000;
                border: 2px solid #667eea;
            }}

            .decision-log h3 {{
                color: #667eea;
                margin-bottom: 10px;
            }}

            .decision-item {{
                padding: 8px;
                margin: 5px 0;
                background: #f8f9fa;
                border-radius: 5px;
                font-size: 0.9em;
                border-left: 3px solid #667eea;
            }}

            .final-section {{
                text-align: center;
                margin-top: 40px;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 8px;
            }}

            .btn-export {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                font-size: 1.1em;
            }}
        </style>
        <script>
            let decisions = [];

            function makeDecision(transistorId, decision) {{
                decisions.push({{
                    transistor_id: transistorId,
                    decision: decision,
                    timestamp: new Date().toISOString()
                }});

                // Update decision log
                updateDecisionLog();

                // Mark conflict as resolved
                const conflictDiv = document.getElementById('conflict-' + transistorId);
                conflictDiv.style.opacity = '0.5';
                conflictDiv.style.pointerEvents = 'none';

                // Show confirmation
                alert('Decision recorded for ' + transistorId + ': ' + decision);
            }}

            function updateDecisionLog() {{
                const logDiv = document.getElementById('decision-log-items');
                logDiv.innerHTML = '';

                decisions.forEach(d => {{
                    const item = document.createElement('div');
                    item.className = 'decision-item';
                    item.innerHTML = `<strong>${{d.transistor_id}}</strong><br><em>${{d.decision}}</em>`;
                    logDiv.appendChild(item);
                }});

                // Update count
                document.getElementById('decision-count').textContent = decisions.length;
            }}

            function exportDecisions() {{
                const json = JSON.stringify({{
                    timestamp: new Date().toISOString(),
                    total_decisions: decisions.length,
                    decisions: decisions
                }}, null, 2);
                const blob = new Blob([json], {{type: 'application/json'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'conflict_resolutions_' + new Date().toISOString().split('T')[0] + '.json';
                a.click();
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Transistor Database Merge Conflicts</h1>
            <p style="color: #666; margin-bottom: 30px;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>

            <div class="summary">
                <h2 style="margin-bottom: 15px;">Summary</h2>
                <div class="summary-stats">
                    <div class="stat">
                        <div class="stat-value">{len(conflicts)}</div>
                        <div class="stat-label">Total Conflicts</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{sum(1 for c in conflicts if c.get('recommendation') == 'keep_new')}</div>
                        <div class="stat-label">Recommend New</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{sum(1 for c in conflicts if c.get('recommendation') == 'keep_existing')}</div>
                        <div class="stat-label">Recommend Existing</div>
                    </div>
                </div>
                <p style="margin-top: 15px; color: #666;">
                    <strong>Instructions:</strong> Review each conflict below and choose which version to keep.
                    The recommendation is based on data completeness scores. Your decisions will be logged
                    and can be exported at the end.
                </p>
            </div>
    """

    # Generate conflict cards
    for i, conflict in enumerate(conflicts, 1):
        transistor_id = conflict['transistor_id']
        existing_score = conflict.get('existing_score', 0)
        new_score = conflict.get('new_score', 0)
        score_diff = conflict.get('score_difference', 0)
        recommendation = conflict.get('recommendation', 'unknown')

        # Load preview data if files exist
        existing_preview = "Preview not available"
        new_preview = "Preview not available"

        existing_file = conflict.get('existing_file', '')
        new_file = conflict.get('new_file', '')

        if Path(existing_file).exists():
            existing_preview = load_transistor_preview(existing_file)

        # For new file, it might be in merged directory
        if Path(new_file).exists():
            new_preview = load_transistor_preview(new_file)

        html += f"""
            <div class="conflict" id="conflict-{transistor_id}">
                <div class="conflict-header">
                    <div class="transistor-id">#{i}: {transistor_id}</div>
                    <div class="score-badge">Score Δ: {score_diff:+.1f}%</div>
                </div>

                <div class="comparison">
                    <div class="version existing">
                        <h3>Existing Version</h3>
                        <div class="file-path">{existing_file}</div>
                        <p><strong>Quality Score:</strong> {existing_score:.1f}%</p>
                        <div style="margin-top: 10px;">
                            {existing_preview}
                        </div>
                    </div>

                    <div class="version new">
                        <h3>New Version</h3>
                        <div class="file-path">{new_file}</div>
                        <p><strong>Quality Score:</strong> {new_score:.1f}%</p>
                        <div style="margin-top: 10px;">
                            {new_preview}
                        </div>
                    </div>
                </div>

                <div class="recommendation">
                    <strong>Recommendation:</strong> {recommendation.replace('_', ' ').title()}
                    ({('New version has higher completeness' if recommendation == 'keep_new' else 'Existing version is more complete')})
                </div>

                <div class="actions">
                    <button class="btn-keep-existing" onclick="makeDecision('{transistor_id}', 'keep_existing')">
                        Keep Existing
                    </button>
                    <button class="btn-keep-new" onclick="makeDecision('{transistor_id}', 'keep_new')">
                        Keep New
                    </button>
                    <button class="btn-merge" onclick="makeDecision('{transistor_id}', 'merge_both')">
                        Merge Both
                    </button>
                    <button class="btn-skip" onclick="makeDecision('{transistor_id}', 'skip')">
                        Skip
                    </button>
                </div>
            </div>
        """

    html += """
            <div class="final-section">
                <h2 style="color: #667eea; margin-bottom: 15px;">Review Complete?</h2>
                <button class="btn-export" onclick="exportDecisions()">
                    Export Decisions
                </button>
            </div>
        </div>

        <div class="decision-log">
            <h3>Decision Log</h3>
            <p style="font-size: 0.9em; color: #667eea; margin-bottom: 10px;">
                <strong id="decision-count">0</strong> decisions made
            </p>
            <div id="decision-log-items">
                <em style="color: #999;">No decisions yet</em>
            </div>
        </div>
    </body>
    </html>
    """

    output_path = Path('/home/tinix/claude_wsl/transistordatabase/CONFLICT_REVIEW.html')
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nConflict review interface generated!")
    print(f"   Open in browser: file://{output_path.absolute()}")
    print(f"   Total conflicts: {len(conflicts)}")
    print(f"   Recommend keeping new: {sum(1 for c in conflicts if c.get('recommendation') == 'keep_new')}")
    print(f"   Recommend keeping existing: {sum(1 for c in conflicts if c.get('recommendation') == 'keep_existing')}\n")

if __name__ == "__main__":
    generate_conflict_review_html()
