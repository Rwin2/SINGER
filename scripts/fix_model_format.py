#!/usr/bin/env python3
"""Fix model.pth format: load state_dict into fresh pilot model and re-save as full model object."""
import sys, os, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sousvide.control.policies.svnet import SVNet

cohort = "ssv_BC_CHUNKED_TEST"
pilot_name = "InstinctJester_chunked"

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
roster_path = os.path.join(workspace, "cohorts", cohort, "roster", pilot_name)

# Load config to create fresh model
config_path = os.path.join(roster_path, "config.json")
with open(config_path) as f:
    config = json.load(f)

# Create fresh SVNet (correct architecture, random weights)
fresh_model = SVNet(config)

for model_file in ["model.pth", "best_model.pth", "last_model.pth"]:
    fpath = os.path.join(roster_path, model_file)
    if not os.path.exists(fpath):
        continue
    obj = torch.load(fpath, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and not hasattr(obj, 'state_dict'):
        # It's a state_dict — load into fresh model and re-save
        fresh_model.load_state_dict(obj)
        torch.save(fresh_model, fpath)
        print(f"Fixed {model_file}: state_dict → full model object")
    else:
        print(f"{model_file}: already correct format ({type(obj).__name__})")
