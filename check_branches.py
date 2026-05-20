import torch
r1 = torch.load("cohorts/SSV_DAGGER_HW1_COMPREHENSIVE/checkpoints/round_1_evaluated.pt", map_location="cpu")
r8 = torch.load("cohorts/SSV_DAGGER_HW1_COMPREHENSIVE/checkpoints/round_8_evaluated.pt", map_location="cpu")
for bid in [64, 86]:
    for d in r1["diagnostics"]:
        if d["object"] == "green clock" and d["branch_id"] == bid:
            print("BC  R1 br%d: success=%s collision=%s goal=%.1f stop=%s" % (bid, d["success"], d["collision"], d["goal_dist"], d["stop_reason"]))
    for d in r8["diagnostics"]:
        if d["object"] == "green clock" and d["branch_id"] == bid:
            print("DAg R8 br%d: success=%s collision=%s goal=%.1f stop=%s" % (bid, d["success"], d["collision"], d["goal_dist"], d["stop_reason"]))
