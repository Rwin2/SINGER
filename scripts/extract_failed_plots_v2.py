"""Extract failed trajectories from DAgger eval HTML plots by parsing plotly JSON."""
import json, re, os, sys

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
    print(f"\nProcessing {filename}...")

    with open(filepath, 'r') as f:
        html = f.read()

    # Plotly HTML stores data as: Plotly.react(gd, {...})
    # or in a <script> with JSON.parse or direct assignment
    # Find the plotly div data — it's typically in window.PlotlyConfig or Plotly.newPlot

    # Try to find the JSON blob between the script tags
    # Pattern: {"data":[...],"layout":{...}}
    match = re.search(r'\{"data":\s*\[', html)
    if not match:
        print(f"  Could not find plotly data in {filename}")
        continue

    # Find the matching end — we need to parse the JSON starting from this point
    start = match.start()
    # Find closing by tracking braces
    depth = 0
    end = start
    for i in range(start, len(html)):
        if html[i] == '{':
            depth += 1
        elif html[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = html[start:end]
    try:
        plotly_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        # Try with a simpler approach — just get the data array
        continue

    traces = plotly_data.get("data", [])
    layout = plotly_data.get("layout", {})
    print(f"  Found {len(traces)} traces")

    # Keep: failed trajectories (COLL/MISS in name) + their expert refs + scene elements
    failed_names = set()
    failed_traces = []
    scene_traces = []  # goal, success zone, point cloud, etc.

    for t in traces:
        name = t.get("name", "") or ""
        if "COLL" in name or "MISS" in name or "FAIL" in name:
            failed_traces.append(t)
            # Extract branch id to also grab the expert trace
            br_match = re.search(r'br(\d+)', name)
            if br_match:
                failed_names.add(f"br{br_match.group(1)}")

    # Grab expert traces for failed branches + scene elements
    for t in traces:
        name = t.get("name", "") or ""
        legendgroup = t.get("legendgroup", "") or ""

        # Expert reference for failed branch
        if "Expert" in name or "expert" in legendgroup:
            for fn in failed_names:
                if fn in name or fn in legendgroup:
                    failed_traces.append(t)
                    break

        # Scene elements (goal, success zone, point cloud, etc.)
        if any(kw in name.lower() for kw in ["goal", "success", "zone", "target", "point cloud"]):
            scene_traces.append(t)

        # Also grab end markers for failed branches
        if not name and legendgroup:
            for fn in failed_names:
                if fn in legendgroup:
                    failed_traces.append(t)
                    break

    all_traces = scene_traces + failed_traces
    n_fail = sum(1 for t in all_traces if any(kw in (t.get("name","") or "") for kw in ["COLL","MISS","FAIL"]))
    print(f"  Keeping {n_fail} failed + {len(scene_traces)} scene + {len(all_traces) - n_fail - len(scene_traces)} related traces")

    # Make all traces visible
    for t in all_traces:
        t["visible"] = True

    layout["title"] = f"<b>Failed Trajectories — {obj_name.replace('_', ' ')}</b> (Round 8)"
    layout.pop("width", None)
    layout.pop("height", None)

    out_html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body {{ margin: 0; }} #plot {{ width: 100vw; height: 100vh; }}</style>
</head>
<body>
    <div id="plot"></div>
    <script>
        var data = {json.dumps(all_traces)};
        var layout = {json.dumps(layout)};
        Plotly.newPlot("plot", data, layout, {{responsive: true}});
    </script>
</body>
</html>"""

    out_path = os.path.join(out_dir, f"failed_R8_{obj_name}.html")
    with open(out_path, 'w') as f:
        f.write(out_html)
    print(f"  -> {out_path} ({len(out_html)//1024}KB)")
