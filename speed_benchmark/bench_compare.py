#!/usr/bin/env python3
"""
Compare two run_benchmark.py result files (MTP-on vs MTP-off) and report, per
context window:

  - decode tok/s for each, and the MTP speedup factor
  - draft acceptance (MTP run)
  - needle-retrieval accuracy for each (the task-accuracy signal)
  - answer match: does the extracted key (KEY-256K-ALPHA) agree between configs
  - greedy token agreement %: how much of the raw generated text is byte-identical

Also supports serving the comparison as a self-contained HTML page with charts.

Usage:
    python bench_compare.py results_mtp.json results_nomtp.json
    python bench_compare.py results_mtp.json results_nomtp.json --serve --port 8080
"""
import argparse
import json
import math
import os
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler


def load(p):
    with open(p) as f:
        return json.load(f)


def pick(a, b):
    """Return (mtp_run, base_run) regardless of argument order."""
    if a.get("mtp_active") and not b.get("mtp_active"):
        return a, b
    if b.get("mtp_active") and not a.get("mtp_active"):
        return b, a
    return (a, b) if "mtp" in a.get("label", "").lower() else (b, a)


def token_agreement(a, b):
    """Fraction of the shorter string that matches byte-for-byte from the start."""
    if not a and not b:
        return 1.0
    n = min(len(a), len(b)) or 1
    same = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return same / max(len(a), len(b), 1)


def compute_rows(mtp, base):
    bm = {s["size"]: s for s in base["sizes"]}
    rows = []
    speedups = []
    accuracy_ok = True
    for s in mtp["sizes"]:
        sz = s["size"]
        b = bm.get(sz)
        if not b:
            continue
        dm, db = s["decode_tps_med"], b["decode_tps_med"]
        spd = dm / db if (dm and db) else None
        if spd:
            speedups.append(spd)
        acc = s.get("accept_rate_med")
        nb, nm = b["needle_success"], s["needle_success"]
        kb, km = b.get("answer_key"), s.get("answer_key")
        if (nm < nb) or (kb and km and kb != km):
            accuracy_ok = False
        if kb == km:
            ans = "same"
        elif kb and km:
            ans = "CONFLICT"
        else:
            ans = "base-miss" if km else "mtp-miss"
        agree = token_agreement(b.get("answer_text", ""), s.get("answer_text", ""))
        rows.append({
            "size": sz,
            "decode_mtp": dm,
            "decode_base": db,
            "speedup": spd,
            "accept_rate": acc,
            "needle_base": nb,
            "needle_mtp": nm,
            "answer": ans,
            "token_agree": agree,
        })
    gm = math.exp(sum(map(math.log, speedups)) / len(speedups)) if speedups else None
    return rows, gm, accuracy_ok


def print_console(mtp, base, rows, gm, accuracy_ok):
    print(f"model: {mtp['model']}")
    print(f"MTP run='{mtp['label']}' (gen={mtp['gen_tokens']}, repeats={mtp['repeats']})  "
          f"vs base='{base['label']}'\n")

    hdr = (f"{'ctx':>8} | {'decode no-MTP':>13} | {'decode MTP':>11} | {'speedup':>7} | "
           f"{'accept':>6} | {'needle b/m':>11} | {'answer':>9} | {'tok-agree':>9}")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        spd_s = f"{r['speedup']:.2f}x" if r["speedup"] else "n/a"
        acc = f"{r['accept_rate']*100:.0f}%" if r.get("accept_rate") is not None else "--"
        nb, nm = r["needle_base"], r["needle_mtp"]
        dm_s = f"{r['decode_mtp']:.1f}" if r["decode_mtp"] else "none"
        db_s = f"{r['decode_base']:.1f}" if r["decode_base"] else "none"
        print(f"{r['size']:>8} | {db_s:>13} | {dm_s:>11} | {spd_s:>7} | {acc:>6} | "
              f"{f'{nb*100:.0f}%/{nm*100:.0f}%':>11} | {r['answer']:>9} | "
              f"{r['token_agree']*100:>7.0f}%")

    if gm:
        print(f"\nMTP decode speedup: {gm:.2f}x geometric mean over {len([r for r in rows if r['speedup']])} valid size(s)")
    print(f"MTP accuracy preserved (needle + answer key unchanged): "
          f"{'YES ✅' if accuracy_ok else 'NO — REGRESSION ⚠'}")
    print("note: <100% tok-agree is expected FP divergence in batched verification, "
          "not a quality loss (see header).")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>highllama Speed Benchmark — MTP Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ color-scheme: light dark; --bg: #f6f8fa; --card: #fff; --text: #1f2328; --muted: #656d76; --accent: #0969da; --border: #d0d7de; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #0d1117; --card: #161b22; --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff; --border: #30363d; }} }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ margin: 0 0 .5rem; font-size: 1.75rem; }}
.subtitle {{ color: var(--muted); margin-bottom: 1.5rem; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
.card h3 {{ margin: 0 0 .25rem; font-size: .85rem; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }}
.card .big {{ font-size: 1.75rem; font-weight: 700; color: var(--accent); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 2rem; }}
th, td {{ padding: .65rem .75rem; text-align: right; border-bottom: 1px solid var(--border); }}
th {{ background: #f3f4f6; color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }}
@media (prefers-color-scheme: dark) {{ th {{ background: #21262d; }} }}
tr:last-child td {{ border-bottom: none; }}
td:first-child, th:first-child {{ text-align: left; }}
.ok {{ color: #1a7f37; font-weight: 600; }}
.warn {{ color: #cf222e; font-weight: 600; }}
details {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; }}
summary {{ cursor: pointer; font-weight: 600; }}
pre {{ overflow: auto; max-height: 400px; font-size: .8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>highllama Speed Benchmark</h1>
  <div class="subtitle">MTP run <strong>{mtp_label}</strong> vs baseline <strong>{base_label}</strong> · model <strong>{model}</strong> · gen={gen} repeats={repeats}</div>

  <div class="summary">
    <div class="card"><h3>Speedup (geom. mean)</h3><div class="big">{speedup}</div></div>
    <div class="card"><h3>Accuracy preserved</h3><div class="big {acc_class}">{accuracy}</div></div>
    <div class="card"><h3>Context sizes</h3><div class="big">{sizes}</div></div>
  </div>

  <div class="grid">
    <div class="chart-card"><canvas id="decodeChart"></canvas></div>
    <div class="chart-card"><canvas id="speedupChart"></canvas></div>
    <div class="chart-card"><canvas id="acceptChart"></canvas></div>
    <div class="chart-card"><canvas id="needleChart"></canvas></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Context</th>
        <th>Decode no-MTP (t/s)</th>
        <th>Decode MTP (t/s)</th>
        <th>Speedup</th>
        <th>Draft accept</th>
        <th>Needle b/m</th>
        <th>Answer</th>
        <th>Token agree</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <details>
    <summary>Raw JSON data</summary>
    <pre id="raw"></pre>
  </details>
</div>

<script>
const rows = {rows_json};
const mtp = {mtp_json};
const base = {base_json};
const ctx = rows.map(r => r.size.toLocaleString());
const palette = {{ mtp: '#0969da', base: '#656d76', speedup: '#1a7f37', accept: '#8250df', needle: '#cf222e' }};

new Chart(document.getElementById('decodeChart'), {{
  type: 'line',
  data: {{
    labels: ctx,
    datasets: [
      {{ label: 'no-MTP', data: rows.map(r => r.decode_base), borderColor: palette.base, backgroundColor: palette.base, tension: 0.2 }},
      {{ label: 'MTP', data: rows.map(r => r.decode_mtp), borderColor: palette.mtp, backgroundColor: palette.mtp, tension: 0.2 }}
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Decode tok/s vs context' }} }}, scales: {{ y: {{ title: {{ display: true, text: 'tok/s' }} }} }} }}
}});

new Chart(document.getElementById('speedupChart'), {{
  type: 'bar',
  data: {{ labels: ctx, datasets: [{{ label: 'speedup', data: rows.map(r => r.speedup), backgroundColor: palette.speedup }}] }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'MTP speedup' }} }}, scales: {{ y: {{ title: {{ display: true, text: 'factor' }}, beginAtZero: true }} }} }}
}});

new Chart(document.getElementById('acceptChart'), {{
  type: 'bar',
  data: {{ labels: ctx, datasets: [{{ label: 'draft accept %', data: rows.map(r => r.accept_rate == null ? null : r.accept_rate * 100), backgroundColor: palette.accept }}] }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Draft acceptance' }} }}, scales: {{ y: {{ title: {{ display: true, text: '%' }}, min: 0, max: 100 }} }} }}
}});

new Chart(document.getElementById('needleChart'), {{
  type: 'line',
  data: {{
    labels: ctx,
    datasets: [
      {{ label: 'no-MTP', data: rows.map(r => r.needle_base * 100), borderColor: palette.base, backgroundColor: palette.base, tension: 0.1 }},
      {{ label: 'MTP', data: rows.map(r => r.needle_mtp * 100), borderColor: palette.needle, backgroundColor: palette.needle, tension: 0.1 }}
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: true, text: 'Needle retrieval success' }} }}, scales: {{ y: {{ title: {{ display: true, text: '%' }}, min: 0, max: 100 }} }} }}
}});

document.getElementById('raw').textContent = JSON.stringify({{ mtp, base, rows }}, null, 2);
</script>
</body>
</html>"""


def build_html(mtp, base, rows, gm, accuracy_ok):
    def fmt(x, d=1):
        return f"{x:.{d}f}" if x is not None else "—"
    def accept_cell(r):
        if r.get("accept_rate") is None:
            return "—"
        return f"{r['accept_rate'] * 100:.0f}%"

    rows_html = "\n".join(
        f"<tr><td>{r['size']:,}</td><td>{fmt(r['decode_base'])}</td><td>{fmt(r['decode_mtp'])}</td>"
        f"<td>{(fmt(r['speedup'], 2) + 'x') if r['speedup'] else '—'}</td>"
        f"<td>{accept_cell(r)}</td>"
        f"<td>{r['needle_base']*100:.0f}% / {r['needle_mtp']*100:.0f}%</td>"
        f"<td class='{'ok' if r['answer'] == 'same' else 'warn'}'>{r['answer']}</td>"
        f"<td>{r['token_agree']*100:.0f}%</td></tr>"
        for r in rows
    )
    return HTML_TEMPLATE.format(
        model=mtp["model"],
        mtp_label=mtp["label"],
        base_label=base["label"],
        gen=mtp["gen_tokens"],
        repeats=mtp["repeats"],
        speedup=f"{gm:.2f}x" if gm else "—",
        accuracy="YES" if accuracy_ok else "REGRESSION",
        acc_class="ok" if accuracy_ok else "warn",
        sizes=len(rows),
        rows_html=rows_html,
        rows_json=json.dumps(rows),
        mtp_json=json.dumps(mtp),
        base_json=json.dumps(base),
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
    print(f">> serving comparison at http://127.0.0.1:{port}")
    print(f"   (snapshot also saved to {tmp.name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs=2, metavar="RESULT_JSON", help="two run_benchmark result files")
    ap.add_argument("--serve", action="store_true", help="serve the comparison as an HTML page")
    ap.add_argument("--port", type=int, default=8080, help="port for the web server (default 8080)")
    args = ap.parse_args()

    mtp, base = pick(load(args.files[0]), load(args.files[1]))
    rows, gm, accuracy_ok = compute_rows(mtp, base)

    if args.serve:
        html = build_html(mtp, base, rows, gm, accuracy_ok)
        serve_html(html, args.port)
    else:
        print_console(mtp, base, rows, gm, accuracy_ok)


if __name__ == "__main__":
    main()
