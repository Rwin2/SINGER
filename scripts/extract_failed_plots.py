"""Extract only failed trajectories from DAgger eval HTML plots and save as new HTMLs."""
import json, re, os

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
    with open(filepath, 'r') as f:
        html = f.read()

    # Extract the plotly JSON data between Plotly.newPlot( and the closing )
    # The data is in the form: Plotly.newPlot("...", data, layout, config)
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]*"\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*,\s*\{', html, re.DOTALL)
    if not match:
        # Try alternate pattern
        match = re.search(r'"data"\s*:\s*(\[.*?\])\s*,\s*"layout"\s*:\s*(\{.*?\})', html, re.DOTALL)

    if not match:
        print(f"Could not parse {filename}, trying different approach...")
        # Extract traces by finding JSON arrays with name fields
        # Find all trace objects
        traces_match = re.search(r'var\s+data\s*=\s*(\[.*?\]);\s*var\s+layout', html, re.DOTALL)
        if not traces_match:
            print(f"  Skipping {filename}")
            continue
        data_str = traces_match.group(1)
        layout_match = re.search(r'var\s+layout\s*=\s*(\{.*?\});\s*Plotly', html, re.DOTALL)
        layout_str = layout_match.group(1) if layout_match else '{}'
    else:
        data_str = match.group(1)
        layout_str = match.group(2)

    try:
        traces = json.loads(data_str)
    except json.JSONDecodeError:
        print(f"  JSON parse failed for {filename}, trying cleanup...")
        # Sometimes plotly uses single quotes or unquoted keys
        continue

    # Filter: keep only traces with COLL or MISS in name, plus goal/expert markers
    failed_traces = []
    other_traces = []  # goal markers, expert paths, etc.
    for t in traces:
        name = t.get("name", "")
        if "COLL" in name or "MISS" in name or "FAIL" in name:
            failed_traces.append(t)
        elif "goal" in name.lower() or "expert" in name.lower() or "target" in name.lower() or "RRT" in name:
            other_traces.append(t)
        elif "Success zone" in name or "success" in name.lower():
            other_traces.append(t)

    all_traces = other_traces + failed_traces
    print(f"{obj_name}: {len(failed_traces)} failed traces (+ {len(other_traces)} reference traces)")

    if not failed_traces:
        print(f"  No failed traces found!")
        continue

    try:
        layout = json.loads(layout_str)
    except:
        layout = {}

    layout["title"] = f"R8 Failed Trajectories — {obj_name.replace('_', ' ')}"

    # Build minimal HTML
    out_html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body {{ margin: 0; }} #plot {{ width: 100vw; height: 100vh; }}</style>
</head>
<body>
    <div id="plot"></div>
    <script>
        var data = {json.dumps(all_traces)};
        var layout = {json.dumps(layout)};
        layout.width = window.innerWidth;
        layout.height = window.innerHeight;
        Plotly.newPlot("plot", data, layout);
    </script>
</body>
</html>"""

    out_path = os.path.join(out_dir, f"failed_R8_{obj_name}.html")
    with open(out_path, 'w') as f:
        f.write(out_html)
    print(f"  -> {out_path}")
