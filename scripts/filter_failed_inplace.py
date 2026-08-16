"""Copy original plotly HTML, hide OK traces + their experts, keep failed experts visible."""
import re, os, json

plots_dir = "cohorts/SSV_DAGGER_COMPREHENSIVE/visualizations/hw1_20260501_102714/plots"
out_dir = "cohorts/SSV_DAGGER_COMPREHENSIVE/visualizations/failed_only"
os.makedirs(out_dir, exist_ok=True)

objects = {
    "green_clock": "Eval_R8_green_clock.html",
    "green_and_pink_leafblower": "Eval_R8_green_and_pink_leafblower.html",
    "yellow_handheld_cordless_drill": "Eval_R8_yellow_handheld_cordless_drill.html",
}

for obj_name, filename in objects.items():
    filepath = os.path.join(plots_dir, filename)
    print(f"Processing {filename}...")

    with open(filepath, 'r') as f:
        html = f.read()

    # Parse traces from the second Plotly.newPlot call
    matches = list(re.finditer(re.escape('Plotly.newPlot('), html))
    search_start = matches[1].end()
    bracket_pos = html.index('[', search_start)

    depth = 0
    end = bracket_pos
    for i in range(bracket_pos, len(html)):
        if html[i] == '[': depth += 1
        elif html[i] == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    traces = json.loads(html[bracket_pos:end])

    # Also get layout
    layout_start = end
    while layout_start < len(html) and html[layout_start] in ' ,\n\r\t':
        layout_start += 1
    brace_depth = 0
    layout_end = layout_start
    for i in range(layout_start, len(html)):
        if html[i] == '{': brace_depth += 1
        elif html[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                layout_end = i + 1
                break
    layout = json.loads(html[layout_start:layout_end])

    # Traces structure: [env, exclusion, success_zone, then per run: expert_blue, run_colored, end_marker]
    # Scene traces = first 3
    scene_traces = traces[:3]
    run_traces = traces[3:]

    # Each run = 3 traces (expert, pilot, end_marker)
    n_runs = len(run_traces) // 3
    print(f"  {n_runs} runs, {len(traces)} total traces")

    # Find failed run indices
    failed_indices = set()
    for run_i in range(n_runs):
        pilot_trace = run_traces[run_i * 3 + 1]
        name = pilot_trace.get("name", "")
        if "COLL" in name or "MISS" in name:
            failed_indices.add(run_i)

    print(f"  Failed runs: {failed_indices}")

    # Build filtered traces: scene + only failed runs (expert + pilot + marker)
    filtered = list(scene_traces)
    for run_i in range(n_runs):
        expert = run_traces[run_i * 3]
        pilot = run_traces[run_i * 3 + 1]
        marker = run_traces[run_i * 3 + 2]

        if run_i in failed_indices:
            # Keep expert visible, make it dashed for distinction
            expert["line"]["dash"] = "dash"
            expert["line"]["width"] = 3
            # Show expert in legend for failed runs
            expert["showlegend"] = True
            pilot_name = pilot.get("name", "")
            br_match = re.search(r'Run (\d+)', pilot_name)
            br_id = br_match.group(1) if br_match else str(run_i)
            expert["name"] = f"Expert br{br_id}"
            filtered.extend([expert, pilot, marker])

    layout["title"]["text"] = f"<b>Failed Trajectories — {obj_name.replace('_', ' ')}</b> (Round 8, {len(failed_indices)} failures)"

    # Use the same plotly.js from the original HTML (embedded)
    # Get everything before the second Plotly.newPlot call
    pre_html = html[:matches[1].start()]
    # Get the config part after layout
    config_start = layout_end
    post_html = html[config_start:]
    # Find the closing of the Plotly.newPlot call in post_html
    # Should be: , {config})  or just )
    paren_match = re.search(r'\)', post_html)
    post_after_call = post_html[paren_match.end():] if paren_match else ""

    # Rebuild
    out_html = (
        pre_html
        + f'Plotly.newPlot("plot", {json.dumps(filtered)}, {json.dumps(layout)}, {{"responsive": true}})'
        + post_after_call
    )

    # Fix the div id to match
    # The original has a UUID div, replace it
    # Find the div id in the original
    div_match = re.search(r'<div id="([^"]+)"', pre_html)
    if div_match:
        original_div = div_match.group(1)
        out_html = out_html.replace('"plot"', f'"{original_div}"', 1)

    out_path = os.path.join(out_dir, f"failed_R8_{obj_name}.html")
    with open(out_path, 'w') as f:
        f.write(out_html)
    print(f"  -> {out_path} ({os.path.getsize(out_path)//1024}KB)")
