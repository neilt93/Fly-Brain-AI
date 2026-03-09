"""
Three-layer tabbed dashboard — Mission Control for plastic fly experiments.

Tab 1: Monitor  — live status, latest run charts, event timeline
Tab 2: Curator  — summary cards, ranked runs, curator events
Tab 3: Analysis — runs table, gait verification, plasticity, compare

Single self-contained HTML file. Dark theme. All plots as base64 PNGs,
all data as inline JSON. No web server, no dependencies.
"""

import json
import base64
import html as html_mod
from pathlib import Path
from datetime import datetime


def _img_to_base64(path: Path) -> str:
    """Read an image file and return base64-encoded data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _esc(text) -> str:
    """HTML-escape a string for safe embedding."""
    return html_mod.escape(str(text))


def _format_val(v, precision=4):
    """Format a value for display."""
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return _esc(str(v))


def _priority_badge(priority: str) -> str:
    """Return colored badge HTML for a curator priority level."""
    colors = {
        "IGNORE": "#555",
        "LOG": "#666",
        "NOTABLE": "#2196F3",
        "ATTENTION": "#FF9800",
        "BREAKTHROUGH": "#f44336",
    }
    color = colors.get(priority, "#888")
    return (f'<span class="badge" style="background:{color}">'
            f'{priority}</span>')


def _card_html(title, data, border_color="#2196F3"):
    """Render a single summary card."""
    if not data:
        return f"""
        <div class="summary-card" style="border-top:3px solid {border_color}">
          <h4>{title}</h4>
          <p class="muted">No data yet</p>
        </div>"""

    run_id = data.get("run_id", "---")
    score = data.get("score", data.get("performance_ratio", "---"))
    explanation = data.get("explanation", "")

    return f"""
    <div class="summary-card" style="border-top:3px solid {border_color}">
      <h4>{title}</h4>
      <div class="card-run-id">{run_id}</div>
      <div class="card-score">{_format_val(score, 3)}</div>
      <p class="card-explanation">{explanation}</p>
    </div>"""


def generate_dashboard(
    experiment_dir: str,
    curator_summary: str = "",
    curator_events: list = None,
    curator_summary_data: dict = None,
    scientist_proposals: list = None,
    dashboard_state: dict = None,
    runs_data: list = None,
    output_path: str = None,
) -> str:
    """Generate three-layer tabbed HTML dashboard.

    Args:
        experiment_dir: path to logs/terrain_shift/
        curator_summary: text summary from CuratorAgent.summarize()
        curator_events: list of CuratorEvent.to_dict() dicts
        curator_summary_data: dict from CuratorAgent.to_summary_json()
        scientist_proposals: list of ExperimentProposal dicts
        dashboard_state: dict from dashboard_state.json (live status)
        runs_data: list of run records from runs.jsonl
        output_path: where to write the HTML

    Returns:
        path to generated HTML file
    """
    exp_dir = Path(experiment_dir)
    if output_path is None:
        output_path = str(exp_dir / "dashboard.html")

    # Load results
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {}

    config = results.get("config", {})
    baseline = results.get("baseline", {})
    shift = results.get("shift", {})
    forgetting = results.get("forgetting", {})
    gait = results.get("gait", {})

    # Load curator events from disk if not provided
    if curator_events is None:
        events_file = exp_dir / "curator" / "events.json"
        if events_file.exists():
            with open(events_file) as f:
                curator_events = json.load(f)
        else:
            curator_events = []

    # Load runs data if not provided
    if runs_data is None:
        runs_file = exp_dir / "logs" / "runs.jsonl"
        if runs_file.exists():
            runs_data = []
            with open(runs_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        runs_data.append(json.loads(line))
        else:
            runs_data = []

    # Load dashboard state if not provided
    if dashboard_state is None:
        state_file = exp_dir / "logs" / "dashboard_state.json"
        if state_file.exists():
            with open(state_file) as f:
                dashboard_state = json.load(f)
        else:
            dashboard_state = {}

    # Load scientist proposals if not provided
    if scientist_proposals is None:
        sci_file = exp_dir / "logs" / "next_experiments.json"
        if sci_file.exists():
            with open(sci_file) as f:
                scientist_proposals = json.load(f)
        else:
            scientist_proposals = []

    # Summary cards
    summary_cards = {}
    if curator_summary_data:
        summary_cards = curator_summary_data.get("summary_cards", {})

    # Encode plots as base64
    plots = {}
    plot_names = [
        "recovery_curves", "comparison", "gait_symmetry", "weight_drift",
        "contact_raster_fixed", "contact_raster_plastic",
        "tripod_score_fixed", "tripod_score_plastic",
    ]
    for name in plot_names:
        img_path = exp_dir / f"{name}.png"
        plots[name] = _img_to_base64(img_path)

    # Build metrics table rows
    b_f = baseline.get("fixed", {})
    b_p = baseline.get("plastic", {})
    s_f = shift.get("fixed", {})
    s_p = shift.get("plastic", {})

    ratio_f = s_f.get("distance", 0) / abs(b_f.get("distance", 1e-9)) if abs(b_f.get("distance", 0)) > 1e-6 else 0
    ratio_p = s_p.get("distance", 0) / abs(b_p.get("distance", 1e-9)) if abs(b_p.get("distance", 0)) > 1e-6 else 0

    winner = ""
    if ratio_p > 0 and ratio_f > 0:
        if ratio_p > ratio_f:
            advantage = ((ratio_p - ratio_f) / ratio_f) * 100
            winner = f"Plastic outperforms fixed by {advantage:.1f}%"
        elif ratio_f > ratio_p:
            advantage = ((ratio_f - ratio_p) / ratio_p) * 100
            winner = f"Fixed outperforms plastic by {advantage:.1f}%"

    # Curator events HTML
    events_rows = ""
    if curator_events:
        for e in curator_events:
            badge = _priority_badge(e.get("priority", "LOG"))
            events_rows += (
                f"<tr>"
                f"<td>{badge}</td>"
                f"<td>{_esc(e.get('change_type', ''))}</td>"
                f"<td>{_esc(e.get('summary', ''))}</td>"
                f"<td>{_esc(e.get('run_id', ''))}</td>"
                f"</tr>\n"
            )
    else:
        events_rows = '<tr><td colspan="4" class="muted" style="text-align:center">No significant events</td></tr>'

    # Runs table rows for Analysis tab
    runs_table_rows = ""
    for run in runs_data:
        m = run.get("metrics", {})
        c = run.get("config", {})
        runs_table_rows += (
            f"<tr>"
            f"<td>{run.get('run_id', '')}</td>"
            f"<td>{c.get('seed', '')}</td>"
            f"<td>{run.get('controller', '')}</td>"
            f"<td>{_format_val(c.get('plastic_lr'), 6)}</td>"
            f"<td>{_format_val(m.get('distance_after'), 4)}</td>"
            f"<td>{_format_val(m.get('gait_symmetry_after'), 4)}</td>"
            f"<td>{_format_val(m.get('performance_ratio'), 4)}</td>"
            f"<td>{_format_val(m.get('recovery_time_steps'))}</td>"
            f"<td>{_format_val(m.get('weight_drift'), 4)}</td>"
            f"<td>{run.get('status', '')}</td>"
            f"</tr>\n"
        )

    # Gait summary for gait sub-tab
    gait_fixed = gait.get("fixed", {})
    gait_plastic = gait.get("plastic", {})

    def gait_row(label, key, explanation=""):
        fv = gait_fixed.get(key, "---")
        pv = gait_plastic.get(key, "---")
        return (f"<tr><td><strong>{label}</strong></td>"
                f"<td style='text-align:right'>{_format_val(fv)}</td>"
                f"<td style='text-align:right'>{_format_val(pv)}</td>"
                f"<td class='muted' style='font-size:0.9em'>{explanation}</td></tr>")

    gait_rows = ""
    gait_rows += gait_row("Tripod Score (mean)", "tripod_score_mean", "0.5-0.8 = good tripod gait")
    gait_rows += gait_row("Stride Symmetry", "stride_symmetry", "1.0 = perfect L/R balance")
    gait_rows += gait_row("Upright Fraction", "upright_fraction", "> 0.9 = stable walking")
    gait_rows += gait_row("Step Freq Mean (Hz)", "step_frequency_mean", "~10-14 Hz = near CPG freq")
    gait_rows += gait_row("Total Drag Events", "total_drag_events", "0 = no dragging")
    gait_rows += gait_row("Total Slip Events", "total_slip_events", "0 = no slipping")
    gait_rows += gait_row("Roll Variance", "roll_var", "Low = stable body")
    gait_rows += gait_row("Pitch Variance", "pitch_var", "Low = stable body")

    # Scientist proposals for Curator tab
    next_rec_html = ""
    if scientist_proposals:
        top = scientist_proposals[0]
        next_rec_html = _card_html(
            "Next Recommendation",
            {
                "run_id": top.get("name", "---"),
                "score": top.get("priority_score", 0),
                "explanation": f"[{top.get('strategy', '')}] {top.get('rationale', '')}",
            },
            border_color="#9C27B0",
        )
    else:
        next_rec_html = _card_html("Next Recommendation", None, border_color="#9C27B0")

    # Monitor status
    status_run_id = dashboard_state.get("run_id", "---")
    status_step = dashboard_state.get("step", "---")
    status_total = dashboard_state.get("total_steps", "---")
    status_phase = dashboard_state.get("phase", "---")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ===== BUILD HTML =====
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Mission Control - Plastic Fly</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
         background: #1a1a2e; color: #e0e0e0; padding: 0; }}

  /* Tabs */
  .tab-bar {{ display: flex; background: #0d1b2a; border-bottom: 2px solid #333; padding: 0 20px; }}
  .tab-btn {{ background: none; border: none; color: #888; padding: 12px 24px; cursor: pointer;
              font-size: 1em; font-family: inherit; border-bottom: 3px solid transparent; }}
  .tab-btn:hover {{ color: #ccc; }}
  .tab-btn.active {{ color: #64b5f6; border-bottom-color: #64b5f6; }}
  .tab-content {{ display: none; padding: 20px; max-width: 1400px; margin: 0 auto; }}
  .tab-content.active {{ display: block; }}

  /* Sub-tabs */
  .subtab-bar {{ display: flex; gap: 2px; margin-bottom: 20px; }}
  .subtab-btn {{ background: #16213e; border: none; color: #888; padding: 8px 16px; cursor: pointer;
                 font-size: 0.9em; font-family: inherit; border-radius: 4px 4px 0 0; }}
  .subtab-btn:hover {{ color: #ccc; }}
  .subtab-btn.active {{ color: #90caf9; background: #1a1a3e; }}
  .subtab-content {{ display: none; }}
  .subtab-content.active {{ display: block; }}

  /* Layout */
  h1 {{ color: #64b5f6; margin-bottom: 5px; padding: 20px 20px 0; }}
  h2 {{ color: #90caf9; margin: 25px 0 12px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  h3 {{ color: #90caf9; margin-bottom: 10px; }}
  .subtitle {{ color: #888; padding: 0 20px 0; margin-bottom: 0; }}

  /* Cards */
  .card {{ background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }}
  .summary-card {{ background: #16213e; border-radius: 8px; padding: 16px; }}
  .summary-card h4 {{ color: #90caf9; font-size: 0.85em; text-transform: uppercase; margin-bottom: 8px; }}
  .card-run-id {{ color: #fff; font-size: 0.85em; margin-bottom: 4px; }}
  .card-score {{ color: #4caf50; font-size: 1.6em; font-weight: bold; margin: 4px 0; }}
  .card-explanation {{ color: #aaa; font-size: 0.8em; line-height: 1.4; }}

  /* Status */
  .status-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .status-item {{ background: #16213e; padding: 12px; border-radius: 6px; text-align: center; }}
  .status-item .label {{ color: #888; font-size: 0.75em; text-transform: uppercase; }}
  .status-item .value {{ color: #fff; font-size: 1.4em; font-weight: bold; margin-top: 4px; }}

  /* Config grid */
  .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }}
  .config-item {{ background: #1a1a3e; padding: 8px 12px; border-radius: 4px; }}
  .config-item .label {{ color: #888; font-size: 0.75em; }}
  .config-item .value {{ color: #fff; font-weight: bold; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #1a1a3e; color: #90caf9; padding: 10px 8px; text-align: left; cursor: pointer;
        font-size: 0.85em; user-select: none; }}
  th:hover {{ background: #222244; }}
  td {{ padding: 8px; border-bottom: 1px solid #2a2a4e; font-size: 0.9em; }}
  tr:hover {{ background: #1a1a3e; }}

  /* Plots */
  .plots-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .plot-card {{ background: #16213e; border-radius: 8px; padding: 15px; }}
  .plot-card img {{ width: 100%; border-radius: 4px; background: #fff; }}
  .plot-card p {{ color: #888; font-size: 0.8em; margin-top: 8px; }}

  /* Misc */
  .badge {{ color: #fff; padding: 2px 8px; border-radius: 3px; font-size: 0.75em; font-weight: bold; }}
  .winner {{ background: #1b5e20; color: #a5d6a7; padding: 12px 20px; border-radius: 6px;
             font-size: 1.1em; font-weight: bold; text-align: center; margin: 15px 0; }}
  .muted {{ color: #666; }}
  .explanation {{ background: #0d1b2a; border-left: 3px solid #64b5f6; padding: 15px; margin: 15px 0;
                  border-radius: 0 6px 6px 0; }}
  .explanation h3 {{ color: #64b5f6; margin-bottom: 8px; }}
  .explanation p {{ color: #aaa; line-height: 1.6; }}
  .event-timeline {{ max-height: 400px; overflow-y: auto; }}

  @media (max-width: 900px) {{ .plots-grid {{ grid-template-columns: 1fr; }}
                               .summary-cards {{ grid-template-columns: 1fr 1fr; }} }}
  @media (max-width: 600px) {{ .summary-cards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Mission Control</h1>
<p class="subtitle">Plastic Fly Experiment Dashboard &mdash; {timestamp}</p>

<!-- TAB BAR -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('monitor', this)">Monitor</button>
  <button class="tab-btn" onclick="switchTab('curator', this)">Curator</button>
  <button class="tab-btn" onclick="switchTab('analysis', this)">Analysis</button>
</div>

<!-- ==================== TAB 1: MONITOR ==================== -->
<div id="tab-monitor" class="tab-content active">

  <h2>Status</h2>
  <div class="status-grid">
    <div class="status-item">
      <div class="label">Run ID</div>
      <div class="value" style="font-size:1em">{status_run_id}</div>
    </div>
    <div class="status-item">
      <div class="label">Step</div>
      <div class="value">{status_step}</div>
    </div>
    <div class="status-item">
      <div class="label">Total Steps</div>
      <div class="value">{status_total}</div>
    </div>
    <div class="status-item">
      <div class="label">Phase</div>
      <div class="value">{status_phase}</div>
    </div>
  </div>

  {"<div class='winner'>" + winner + "</div>" if winner else ""}

  <h2>Latest Run</h2>
  <div class="plots-grid">
    <div class="plot-card">
      <h3>Recovery Curves</h3>
      {"<img src='" + plots['recovery_curves'] + "' alt='Recovery curves'/>" if plots.get('recovery_curves') else "<p class='muted'>Not generated</p>"}
      <p>Forward velocity through flat, blocks, flat. Plastic should recover faster.</p>
    </div>
    <div class="plot-card">
      <h3>Weight Drift</h3>
      {"<img src='" + plots['weight_drift'] + "' alt='Weight drift'/>" if plots.get('weight_drift') else "<p class='muted'>Not generated</p>"}
      <p>Plastic recurrent weight change from init. Spikes on blocks = adaptation.</p>
    </div>
  </div>

  <h2>Results</h2>
  <div class="card">
    <table>
      <tr>
        <th>Metric</th>
        <th style="text-align:right;color:#2196F3">Fixed</th>
        <th style="text-align:right;color:#FF5722">Plastic</th>
        <th>Meaning</th>
      </tr>
      <tr><td><strong>Flat distance (mm)</strong></td>
          <td style="text-align:right">{_format_val(b_f.get("distance"))}</td>
          <td style="text-align:right">{_format_val(b_p.get("distance"))}</td>
          <td class="muted">Baseline walking on flat ground</td></tr>
      <tr><td><strong>Flat symmetry</strong></td>
          <td style="text-align:right">{_format_val(b_f.get("symmetry"))}</td>
          <td style="text-align:right">{_format_val(b_p.get("symmetry"))}</td>
          <td class="muted">L/R stepping balance. 1.0 = perfect</td></tr>
      <tr><td><strong>Blocks distance (mm)</strong></td>
          <td style="text-align:right">{_format_val(s_f.get("distance"))}</td>
          <td style="text-align:right">{_format_val(s_p.get("distance"))}</td>
          <td class="muted">Distance on uneven terrain</td></tr>
      <tr><td><strong>Blocks symmetry</strong></td>
          <td style="text-align:right">{_format_val(s_f.get("symmetry"))}</td>
          <td style="text-align:right">{_format_val(s_p.get("symmetry"))}</td>
          <td class="muted">Stepping balance on blocks</td></tr>
      <tr><td><strong>Perf ratio</strong></td>
          <td style="text-align:right">{_format_val(ratio_f)}</td>
          <td style="text-align:right">{_format_val(ratio_p)}</td>
          <td class="muted">blocks/flat distance. Higher = better adaptation</td></tr>
      <tr><td><strong>Weight drift</strong></td>
          <td style="text-align:right">---</td>
          <td style="text-align:right">{_format_val(s_p.get("weight_drift"))}</td>
          <td class="muted">L2 change in plastic weights</td></tr>
    </table>
  </div>

  <h2>Event Timeline</h2>
  <div class="card event-timeline">
    <table>
      <tr><th>Priority</th><th>Type</th><th>Summary</th><th>Run</th></tr>
      {events_rows}
    </table>
  </div>
</div>

<!-- ==================== TAB 2: CURATOR ==================== -->
<div id="tab-curator" class="tab-content">

  <h2>Summary Cards</h2>
  <div class="summary-cards">
    {_card_html("Best Run", summary_cards.get("best_run"), "#4caf50")}
    {_card_html("Most Surprising", summary_cards.get("most_surprising"), "#ff9800")}
    {_card_html("Biggest Failure", summary_cards.get("biggest_failure"), "#f44336")}
    {_card_html("Most Demo-Worthy", summary_cards.get("most_demo_worthy"), "#2196F3")}
    {next_rec_html}
  </div>

  <h2>Curator Events</h2>
  <div class="explanation">
    <h3>What is the Curator?</h3>
    <p>The curator watches experiments and decides what matters.
       Priority levels: IGNORE &lt; LOG &lt; NOTABLE &lt; ATTENTION &lt; BREAKTHROUGH.
       ATTENTION+ events are things to look at. BREAKTHROUGH = stop and look now.</p>
  </div>
  <div class="card">
    <table>
      <tr><th>Priority</th><th>Type</th><th>Summary</th><th>Run</th></tr>
      {events_rows}
    </table>
  </div>

  {"<h2>Scientist Proposals</h2><div class='card'><table><tr><th>Name</th><th>Strategy</th><th>Score</th><th>Rationale</th></tr>" + "".join(f"<tr><td>{_esc(p.get('name',''))}</td><td>{_esc(p.get('strategy',''))}</td><td>{_format_val(p.get('priority_score'), 2)}</td><td>{_esc(p.get('rationale',''))}</td></tr>" for p in scientist_proposals) + "</table></div>" if scientist_proposals else ""}
</div>

<!-- ==================== TAB 3: ANALYSIS ==================== -->
<div id="tab-analysis" class="tab-content">

  <!-- Sub-tab bar -->
  <div class="subtab-bar">
    <button class="subtab-btn active" onclick="switchSubTab('overview', this)">Overview</button>
    <button class="subtab-btn" onclick="switchSubTab('runs', this)">Runs Table</button>
    <button class="subtab-btn" onclick="switchSubTab('gait', this)">Gait</button>
    <button class="subtab-btn" onclick="switchSubTab('plasticity', this)">Plasticity</button>
  </div>

  <!-- Sub-tab: Overview -->
  <div id="subtab-overview" class="subtab-content active">
    <h2>Experiment Config</h2>
    <div class="card">
      <div class="config-grid">
        <div class="config-item"><div class="label">Total Steps</div><div class="value">{config.get('total_steps', '?')}</div></div>
        <div class="config-item"><div class="label">Flat Section</div><div class="value">{config.get('flat_length', '?')} mm</div></div>
        <div class="config-item"><div class="label">Blocks Section</div><div class="value">{config.get('blocks_length', '?')} mm</div></div>
        <div class="config-item"><div class="label">Block Height</div><div class="value">{config.get('height_range', '?')}</div></div>
        <div class="config-item"><div class="label">CPG Frequency</div><div class="value">{config.get('cpg_freq', '?')} Hz</div></div>
        <div class="config-item"><div class="label">Modulation Scale</div><div class="value">{config.get('modulation_scale', '?')}</div></div>
        <div class="config-item"><div class="label">Plastic LR</div><div class="value">{config.get('plastic_lr', '?')}</div></div>
        <div class="config-item"><div class="label">Weight Decay</div><div class="value">{config.get('plastic_decay', '?')}</div></div>
        <div class="config-item"><div class="label">Hidden Dim</div><div class="value">{config.get('hidden_dim', '?')}</div></div>
        <div class="config-item"><div class="label">Sparsity</div><div class="value">{config.get('sparsity', '?')}</div></div>
        <div class="config-item"><div class="label">Seed</div><div class="value">{config.get('seed', '?')}</div></div>
        <div class="config-item"><div class="label">Timestep</div><div class="value">{config.get('timestep', '?')} s</div></div>
      </div>
    </div>

    <div class="plots-grid">
      <div class="plot-card">
        <h3>Performance Comparison</h3>
        {"<img src='" + plots['comparison'] + "' alt='Comparison'/>" if plots.get('comparison') else "<p class='muted'>Not generated</p>"}
      </div>
      <div class="plot-card">
        <h3>Gait Symmetry Over Time</h3>
        {"<img src='" + plots['gait_symmetry'] + "' alt='Gait symmetry'/>" if plots.get('gait_symmetry') else "<p class='muted'>Not generated</p>"}
      </div>
    </div>
  </div>

  <!-- Sub-tab: Runs Table -->
  <div id="subtab-runs" class="subtab-content">
    <h2>All Runs</h2>
    <div class="card">
      <table id="runs-table">
        <thead>
          <tr>
            <th onclick="sortTable(0)">Run ID</th>
            <th onclick="sortTable(1)">Seed</th>
            <th onclick="sortTable(2)">Controller</th>
            <th onclick="sortTable(3)">LR</th>
            <th onclick="sortTable(4)">Distance</th>
            <th onclick="sortTable(5)">Symmetry</th>
            <th onclick="sortTable(6)">Perf Ratio</th>
            <th onclick="sortTable(7)">Recovery</th>
            <th onclick="sortTable(8)">Drift</th>
            <th onclick="sortTable(9)">Status</th>
          </tr>
        </thead>
        <tbody>
          {runs_table_rows if runs_table_rows else '<tr><td colspan="10" class="muted" style="text-align:center">No runs recorded yet</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Sub-tab: Gait -->
  <div id="subtab-gait" class="subtab-content">
    <h2>Gait Verification</h2>
    <div class="explanation">
      <h3>Is the fly actually walking?</h3>
      <p>Contact rasters show stance (dark) / swing (light) per leg over time.
         A healthy tripod gait shows alternating bands. Tripod score measures
         anti-phase coordination between the two leg groups. Step frequency should
         match CPG frequency (~{config.get('cpg_freq', 12)} Hz).</p>
    </div>

    <div class="card">
      <table>
        <tr><th>Metric</th>
            <th style="text-align:right;color:#2196F3">Fixed</th>
            <th style="text-align:right;color:#FF5722">Plastic</th>
            <th>Interpretation</th></tr>
        {gait_rows}
      </table>
    </div>

    <div class="plots-grid">
      <div class="plot-card">
        <h3>Contact Raster (Fixed)</h3>
        {"<img src='" + plots['contact_raster_fixed'] + "' alt='Raster fixed'/>" if plots.get('contact_raster_fixed') else "<p class='muted'>Not generated</p>"}
      </div>
      <div class="plot-card">
        <h3>Contact Raster (Plastic)</h3>
        {"<img src='" + plots['contact_raster_plastic'] + "' alt='Raster plastic'/>" if plots.get('contact_raster_plastic') else "<p class='muted'>Not generated</p>"}
      </div>
      <div class="plot-card">
        <h3>Tripod Score (Fixed)</h3>
        {"<img src='" + plots['tripod_score_fixed'] + "' alt='Tripod fixed'/>" if plots.get('tripod_score_fixed') else "<p class='muted'>Not generated</p>"}
      </div>
      <div class="plot-card">
        <h3>Tripod Score (Plastic)</h3>
        {"<img src='" + plots['tripod_score_plastic'] + "' alt='Tripod plastic'/>" if plots.get('tripod_score_plastic') else "<p class='muted'>Not generated</p>"}
      </div>
    </div>
  </div>

  <!-- Sub-tab: Plasticity -->
  <div id="subtab-plasticity" class="subtab-content">
    <h2>Plasticity Analysis</h2>
    <div class="plots-grid">
      <div class="plot-card">
        <h3>Weight Drift Over Time</h3>
        {"<img src='" + plots['weight_drift'] + "' alt='Weight drift'/>" if plots.get('weight_drift') else "<p class='muted'>Not generated</p>"}
        <p>L2 norm of recurrent weight change from initialization.
           Controlled drift with homeostatic decay indicates stable learning.</p>
      </div>
      <div class="plot-card">
        <h3>Recovery Curves</h3>
        {"<img src='" + plots['recovery_curves'] + "' alt='Recovery'/>" if plots.get('recovery_curves') else "<p class='muted'>Not generated</p>"}
        <p>Velocity over time. If plastic recovers faster on blocks,
           plasticity is doing useful work.</p>
      </div>
    </div>

    <div class="explanation">
      <h3>Architecture</h3>
      <p><strong>Rule:</strong> <code>dw = lr * (pre * post / hidden_dim) - lr * decay * (w - w_init)</code><br>
         Hebbian outer product drives adaptation. Homeostatic decay prevents runaway.
         Plasticity cap clips weights to [-0.5, 0.5].</p>
    </div>
  </div>
</div>

<!-- ==================== JAVASCRIPT ==================== -->
<script>
function switchTab(tabId, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  btn.classList.add('active');
}}

function switchSubTab(subId, btn) {{
  document.querySelectorAll('.subtab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.subtab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('subtab-' + subId).classList.add('active');
  btn.classList.add('active');
}}

function sortTable(colIdx) {{
  const table = document.getElementById('runs-table');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));

  if (rows.length <= 1 && rows[0].cells.length <= 1) return;

  const dir = table.dataset.sortCol == colIdx && table.dataset.sortDir == 'asc' ? 'desc' : 'asc';
  table.dataset.sortCol = colIdx;
  table.dataset.sortDir = dir;

  rows.sort((a, b) => {{
    let va = a.cells[colIdx]?.textContent.trim() || '';
    let vb = b.cells[colIdx]?.textContent.trim() || '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) {{
      return dir === 'asc' ? na - nb : nb - na;
    }}
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  }});

  rows.forEach(row => tbody.appendChild(row));
}}
</script>

</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
