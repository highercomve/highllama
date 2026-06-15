#!/usr/bin/env python3
"""
Leaderboard across one or more code-benchmark result files, including both
single-shot (run_code_benchmark.py) and agentic (run_agentic_benchmark.py) runs.

Reads results/**/*.json (or the paths you pass), groups results by base model,
and shows how each model scores per agent/backend so you can compare, e.g.,
opencode-agent vs pi-agent vs single-shot for the same model.

Also supports serving the comparison as a self-contained HTML page with charts.

Usage:
    python score_compare.py                       # everything in results/
    python score_compare.py results/a.json results/b.json
    python score_compare.py --serve --port 8080
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROVIDER_PREFIXES = ("opencode-go/", "llamacpp/", "anthropic/", "openai/")


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def normalize_base_model(blob):
    """Strip provider prefixes and agent suffixes so the same base model groups together."""
    name = blob.get("agent_model") or blob["model"]
    for p in PROVIDER_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
    name = re.sub(r"\s+\([^)]*-agent\)$", "", name)
    return name


def run_label(blob):
    """Label used to distinguish single-shot from each agent backend."""
    mode = blob.get("mode", "")
    if mode.endswith("-agent"):
        return mode
    return "single-shot"


def group_by_model(blobs):
    groups = defaultdict(list)
    for b in blobs:
        groups[normalize_base_model(b)].append(b)
    return groups


def console_report(blobs):
    groups = group_by_model(blobs)
    print(f"Models: {len(groups)}  |  total result files: {len(blobs)}\n")

    for base in sorted(groups):
        print(base)
        runs = sorted(groups[base], key=lambda b: (run_label(b), b["summary"]["pass_at_1_pct"]), reverse=False)
        # sort single-shot first, then agents alphabetically
        runs = sorted(groups[base], key=lambda b: (0 if run_label(b) == "single-shot" else 1, run_label(b)))
        for b in runs:
            s = b["summary"]
            label = run_label(b)
            tps = s.get("avg_tok_s")
            print(f"  {label:22}  pass@1 {s['pass_at_1_pct']:>5}%  test {s['test_pass_pct']:>5}%  "
                  f"tasks {s['tasks']:>3}  {s.get('avg_gen_s', 0):>6.2f}s/task  "
                  f"tok/s {tps if tps else '-':>6}")

    print("\n(pass@1 = task fully correct; test% = fraction of hidden checks passed; "
          "s/task = avg model latency)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>highllama Code Benchmark — Model vs Agent</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ color-scheme: light dark; --bg: #f6f8fa; --card: #fff; --text: #1f2328; --muted: #656d76; --accent: #0969da; --border: #d0d7de; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #0d1117; --card: #161b22; --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff; --border: #30363d; }} }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ margin: 0 0 .5rem; font-size: 1.75rem; }}
.subtitle {{ color: var(--muted); margin-bottom: 1.5rem; }}
.tabs {{ display: flex; gap: .5rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); }}
.tab {{ padding: .6rem 1rem; cursor: pointer; border: none; background: transparent; color: var(--muted); font-weight: 600; border-bottom: 2px solid transparent; }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 2rem; }}
th, td {{ padding: .55rem .65rem; text-align: right; border-bottom: 1px solid var(--border); font-size: .9rem; }}
th {{ background: #f3f4f6; color: var(--muted); font-weight: 600; font-size: .75rem; text-transform: uppercase; letter-spacing: .03em; position: sticky; top: 0; }}
@media (prefers-color-scheme: dark) {{ th {{ background: #21262d; }} }}
tr:last-child td {{ border-bottom: none; }}
td:first-child, th:first-child {{ text-align: left; font-weight: 600; }}
td.lang-cell {{ text-align: center; }}
.bar {{ display: inline-block; height: 8px; border-radius: 4px; background: var(--accent); vertical-align: middle; margin-right: .35rem; }}
.ok {{ color: #1a7f37; }}
.miss {{ color: #cf222e; }}
.agent-single {{ color: #0969da; }}
.agent-opencode {{ color: #1a7f37; }}
.agent-pi {{ color: #8250df; }}
.badge {{ display: inline-block; padding: .15rem .4rem; border-radius: 999px; font-size: .75rem; font-weight: 600; background: #ddf4ff; color: #0969da; }}
@media (prefers-color-scheme: dark) {{ .badge {{ background: #0c2d6b; color: #a5d6ff; }} }}
details {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }}
summary {{ cursor: pointer; font-weight: 600; }}
pre {{ overflow: auto; max-height: 500px; font-size: .8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>highllama Code Benchmark</h1>
  <div class="subtitle">{count} result file(s) · {model_count} base model(s) · comparing single-shot vs agentic backends</div>

  <div class="tabs">
    <button class="tab active" onclick="showPanel('compare')">Model vs Agent</button>
    <button class="tab" onclick="showPanel('leaderboard')">Raw Leaderboard</button>
    <button class="tab" onclick="showPanel('raw')">Raw JSON</button>
  </div>

  <div id="compare" class="panel active">
    <div class="grid">
      <div class="card"><canvas id="comparePassChart"></canvas></div>
      <div class="card"><canvas id="compareTestChart"></div>
      <div class="card"><canvas id="compareLatencyChart"></canvas></div>
      <div class="card"><canvas id="compareLangChart"></canvas></div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Backend</th>
          <th>pass@1</th>
          <th>test%</th>
          <th>tasks</th>
          <th>s/task</th>
          <th>tok/s</th>
        </tr>
      </thead>
      <tbody>{compare_rows_html}</tbody>
    </table>
  </div>

  <div id="leaderboard" class="panel">
    <div class="grid">
      <div class="card"><canvas id="passChart"></canvas></div>
      <div class="card"><canvas id="testChart"></canvas></div>
      <div class="card"><canvas id="latencyChart"></canvas></div>
      <div class="card"><canvas id="langChart"></canvas></div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Backend</th>
          <th>pass@1</th>
          <th>test%</th>
          <th>tasks</th>
          <th>s/task</th>
          <th>tok/s</th>
          {lang_headers}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div id="raw" class="panel">
    <pre id="rawJson"></pre>
  </div>
</div>

<script>
function showPanel(id) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}

const blobs = {blobs_json};
const langs = {langs_json};
const palette = ['#0969da', '#1a7f37', '#8250df', '#cf222e', '#fb8500', '#219ebc', '#ff006e', '#8338ec'];

const comparison = {comparison_json};
const compareModels = comparison.map(c => c.model);
const compareBackends = {compare_backends_json};

new Chart(document.getElementById('comparePassChart'), {{
  type: 'bar',
  data: {{
    labels: compareModels,
    datasets: compareBackends.map((b, i) => ({{
      label: b,
      data: comparison.map(c => c[b]?.pass_at_1_pct ?? null),
      backgroundColor: palette[i % palette.length]
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'pass@1 by backend' }} }}, scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: '%' }} }} }} }}
}});

new Chart(document.getElementById('compareTestChart'), {{
  type: 'bar',
  data: {{
    labels: compareModels,
    datasets: compareBackends.map((b, i) => ({{
      label: b,
      data: comparison.map(c => c[b]?.test_pass_pct ?? null),
      backgroundColor: palette[i % palette.length]
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'test-pass % by backend' }} }}, scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: '%' }} }} }} }}
}});

new Chart(document.getElementById('compareLatencyChart'), {{
  type: 'bar',
  data: {{
    labels: compareModels,
    datasets: compareBackends.map((b, i) => ({{
      label: b,
      data: comparison.map(c => c[b]?.avg_gen_s ?? null),
      backgroundColor: palette[i % palette.length]
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'avg latency by backend (s/task)' }} }} }}
}});

new Chart(document.getElementById('compareLangChart'), {{
  type: 'bar',
  data: {{
    labels: compareModels,
    datasets: langs.map((l, i) => ({{
      label: l,
      data: comparison.map(c => {{ const v = c['single-shot']; if (!v) return null; const cell = v.by_language[l]; return cell ? cell.pass_at_1_pct : null; }}),
      backgroundColor: palette[i % palette.length]
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'single-shot pass@1 by language' }} }}, scales: {{ y: {{ min: 0, max: 100 }} }} }}
}});

const labels = blobs.map(b => b.model);
new Chart(document.getElementById('passChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'pass@1 %', data: blobs.map(b => b.summary.pass_at_1_pct), backgroundColor: palette[0] }}] }},
  options: {{ indexAxis: 'y', responsive: true, plugins: {{ title: {{ display: true, text: 'pass@1' }} }}, scales: {{ x: {{ min: 0, max: 100 }} }} }}
}});

new Chart(document.getElementById('testChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'test-pass %', data: blobs.map(b => b.summary.test_pass_pct), backgroundColor: palette[1] }}] }},
  options: {{ indexAxis: 'y', responsive: true, plugins: {{ title: {{ display: true, text: 'test-pass %' }} }}, scales: {{ x: {{ min: 0, max: 100 }} }} }}
}});

new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 's/task', data: blobs.map(b => b.summary.avg_gen_s || 0), backgroundColor: palette[2] }}] }},
  options: {{ indexAxis: 'y', responsive: true, plugins: {{ title: {{ display: true, text: 'avg latency (s/task)' }} }} }}
}});

new Chart(document.getElementById('langChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: langs.map((l, i) => ({{
      label: l,
      data: blobs.map(b => {{ const c = b.summary.by_language[l]; return c ? c.pass_at_1_pct : 0; }}),
      backgroundColor: palette[i % palette.length]
    }}))
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'pass@1 by language' }} }}, scales: {{ y: {{ min: 0, max: 100, title: {{ display: true, text: '%' }} }} }} }}
}});

document.getElementById('rawJson').textContent = JSON.stringify(blobs, null, 2);
</script>
</body>
</html>"""


def build_html(blobs):
    blobs = sorted(blobs, key=lambda b: b["summary"]["pass_at_1_pct"], reverse=True)
    langs = sorted({l for b in blobs for l in b["summary"]["by_language"]})
    groups = group_by_model(blobs)

    def lang_cell(b, l):
        c = b["summary"]["by_language"].get(l)
        if not c:
            return "<td class='lang-cell'>—</td>"
        pct = c["pass_at_1_pct"]
        color = "ok" if pct >= 50 else "miss"
        return f"<td class='lang-cell {color}'>{pct}%</td>"

    rows_html = "\n".join(
        f"<tr><td>{b['model']}</td><td><span class='badge'>{run_label(b)}</span></td>"
        f"<td><span class='bar' style='width:{b['summary']['pass_at_1_pct'] * .6}px'></span>{b['summary']['pass_at_1_pct']}%</td>"
        f"<td>{b['summary']['test_pass_pct']}%</td>"
        f"<td>{b['summary']['tasks']}</td>"
        f"<td>{b['summary'].get('avg_gen_s', 0):.2f}</td>"
        f"<td>{b['summary'].get('avg_tok_s', '—')}</td>"
        + "".join(lang_cell(b, l) for l in langs)
        + "</tr>"
        for b in blobs
    )

    # Build comparison data: for each base model, pick the best run per backend.
    comparison = []
    all_backends = set()
    for base in sorted(groups):
        entry = {"model": base}
        for b in groups[base]:
            label = run_label(b)
            all_backends.add(label)
            if label not in entry or b["summary"]["pass_at_1_pct"] > entry[label]["pass_at_1_pct"]:
                entry[label] = {
                    "pass_at_1_pct": b["summary"]["pass_at_1_pct"],
                    "test_pass_pct": b["summary"]["test_pass_pct"],
                    "tasks": b["summary"]["tasks"],
                    "avg_gen_s": b["summary"].get("avg_gen_s", 0),
                    "avg_tok_s": b["summary"].get("avg_tok_s"),
                    "by_language": b["summary"]["by_language"],
                }
        comparison.append(entry)
    # stable ordering: single-shot first, then agents alphabetically
    backend_order = sorted(all_backends, key=lambda x: (0 if x == "single-shot" else 1, x))

    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.1f}"
        return str(x)

    compare_rows_html = "\n".join(
        f"<tr><td rowspan='{len(backend_order)}'>{c['model']}</td>"
        f"<td>{backend_order[0]}</td>"
        f"<td>{fmt(c.get(backend_order[0], {}).get('pass_at_1_pct', '—'))}</td>"
        f"<td>{fmt(c.get(backend_order[0], {}).get('test_pass_pct', '—'))}</td>"
        f"<td>{fmt(c.get(backend_order[0], {}).get('tasks', '—'))}</td>"
        f"<td>{fmt(c.get(backend_order[0], {}).get('avg_gen_s', '—'))}</td>"
        f"<td>{fmt(c.get(backend_order[0], {}).get('avg_tok_s', '—'))}</td></tr>"
        + "".join(
            f"<tr><td>{b}</td>"
            f"<td>{fmt(c.get(b, {}).get('pass_at_1_pct', '—'))}</td>"
            f"<td>{fmt(c.get(b, {}).get('test_pass_pct', '—'))}</td>"
            f"<td>{fmt(c.get(b, {}).get('tasks', '—'))}</td>"
            f"<td>{fmt(c.get(b, {}).get('avg_gen_s', '—'))}</td>"
            f"<td>{fmt(c.get(b, {}).get('avg_tok_s', '—'))}</td></tr>"
            for b in backend_order[1:]
        )
        for c in comparison
    )

    return HTML_TEMPLATE.format(
        count=len(blobs),
        model_count=len(groups),
        lang_headers="".join(f"<th>{l}</th>" for l in langs),
        rows_html=rows_html,
        compare_rows_html=compare_rows_html,
        blobs_json=json.dumps(blobs),
        langs_json=json.dumps(langs),
        comparison_json=json.dumps(comparison),
        compare_backends_json=json.dumps(backend_order),
    )


class HtmlHandler(BaseHTTPRequestHandler):
    def __init__(self, html, *args, **kwargs):
        self.html = html
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.html.encode("utf-8"))

    def log_message(self, fmt, *args):
        pass


def serve_html(html, port):
    server = HTTPServer(("127.0.0.1", port), lambda *args, **kwargs: HtmlHandler(html, *args, **kwargs))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False)
    tmp.write(html)
    tmp.close()
    print(f">> serving leaderboard at http://127.0.0.1:{port}")
    print(f"   (snapshot also saved to {tmp.name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", metavar="RESULT_JSON", help="result files (default: all results/**/*.json)")
    ap.add_argument("--serve", action="store_true", help="serve the comparison as an HTML page")
    ap.add_argument("--port", type=int, default=8080, help="port for the web server (default 8080)")
    args = ap.parse_args()

    paths = args.files or sorted(glob.glob(os.path.join(HERE, "results", "**", "*.json"), recursive=True))
    if not paths:
        sys.exit("!! no result files (run run_code_benchmark.py first)")
    blobs = load(paths)

    if args.serve:
        html = build_html(blobs)
        serve_html(html, args.port)
    else:
        console_report(blobs)


if __name__ == "__main__":
    main()
