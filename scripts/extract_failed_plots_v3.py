"""Extract failed trajectories from DAgger eval HTML plots."""
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
    print(f"\nProcessing {filename}...")

    with open(filepath, 'r') as f:
        html = f.read()

    # Find the second Plotly.newPlot which has the actual plot data
    # Format: Plotly.newPlot( "uuid", [{traces}], {layout}, {config})
    matches = list(re.finditer(re.escape('Plotly.newPlot('), html))
    if len(matches) < 2:
        print(f"  Could not find second Plotly.newPlot, found {len(matches)}")
        continue

    # From the second match, find the first '[' which starts the traces array
    search_start = matches[1].end()
    bracket_pos = html.index('[', search_start)
    data_start = bracket_pos

    # Parse the traces array by finding matching brackets
    bracket_depth = 0
    traces_end = data_start
    for i in range(data_start, len(html)):
        if html[i] == '[':
            bracket_depth += 1
        elif html[i] == ']':
            bracket_depth -= 1
            if bracket_depth == 0:
                traces_end = i + 1
                break

    traces_json = html[data_start:traces_end]

    # Now find the layout object after the traces
    # Skip comma and whitespace
    layout_start = traces_end
    while layout_start < len(html) and html[layout_start] in ' ,\n\r\t':
        layout_start += 1

    brace_depth = 0
    layout_end = layout_start
    for i in range(layout_start, len(html)):
        if html[i] == '{':
            brace_depth += 1
        elif html[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                layout_end = i + 1
                break

    layout_json = html[layout_start:layout_end]

    try:
        traces = json.loads(traces_json)
        layout = json.loads(layout_json)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        continue

    print(f"  Parsed {len(traces)} traces")

    # Identify failed trajectory traces and their related traces
    failed_branch_ids = set()
    failed_traces = []
    scene_traces = []

    for t in traces:
        name = t.get("name", "") or ""
        if "COLL" in name or "MISS" in name:
            failed_traces.append(t)
            br = re.search(r'Run (\d+)', name)
            if br:
                failed_branch_ids.add(int(br.group(1)))

    # Grab scene elements (first few traces are usually point cloud, goal marker, success zone)
    for t in traces:
        name = t.get("name", "") or ""
        legendgroup = t.get("legendgroup", "") or ""
        # Scene elements: point cloud (no name), goal/target markers, success zone
        if "Success Zone" in name or "Goal" in name or "target" in name.lower():
            scene_traces.append(t)
        # End markers for failed runs (no name, has legendgroup matching)
        elif not name and legendgroup:
            run_match = re.search(r'(\d+)', legendgroup)
            if run_match and int(run_match.group(1)) in failed_branch_ids:
                failed_traces.append(t)

    # Also grab expert references for failed runs
    for t in traces:
        name = t.get("name", "") or ""
        legendgroup = t.get("legendgroup", "") or ""
        if "Expert" in name or "expert" in legendgroup.lower():
            # Check if it's for a failed branch
            for bid in failed_branch_ids:
                if str(bid) in legendgroup or f"Run {bid}" in name:
                    t_copy = dict(t)
                    t_copy["visible"] = True
                    failed_traces.append(t_copy)
                    break

    # Also grab the point cloud (usually first trace, large scatter3d with many points)
    for t in traces[:5]:
        if t.get("type") == "scatter3d" and "marker" in t:
            marker = t["marker"]
            if isinstance(marker.get("color"), list) and len(marker["color"]) > 100:
                scene_traces.append(t)
                break

    all_traces = scene_traces + failed_traces
    n_fail = sum(1 for t in all_traces if any(kw in (t.get("name","") or "") for kw in ["COLL","MISS"]))
    print(f"  {n_fail} failed trajectories + {len(scene_traces)} scene + {len(failed_traces)-n_fail} markers")

    layout["title"] = {"text": f"<b>Failed Trajectories — {obj_name.replace('_', ' ')}</b> (Round 8, {n_fail} failures)"}
    layout.pop("width", None)
    layout.pop("height", None)

    for t in all_traces:
        t["visible"] = True

    # Use plotly.js CDN
    out_html = f"""<html>
<head><meta charset="utf-8" /></head>
<body>
    <div id="plot" style="width:100vw;height:100vh;"></div>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script>
        Plotly.newPlot("plot", {json.dumps(all_traces)}, {json.dumps(layout)}, {{responsive: true}});
    </script>
</body>
</html>"""

    out_path = os.path.join(out_dir, f"failed_R8_{obj_name}.html")
    with open(out_path, 'w') as f:
        f.write(out_html)
    print(f"  -> {out_path} ({os.path.getsize(out_path)//1024}KB)")
