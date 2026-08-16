"""Test whether the trained Commander is sensitive to obj_com (centroid) input."""
import torch
import numpy as np

model = torch.load('cohorts/ssv_BC_GT_CENTROID_V3/roster/InstinctJester_gt/model.pth', map_location='cpu', weights_only=False)
model.eval()

commander = model.network['CommanderSV']
hist_enc = model.network['HistoryEncoder']
vis_mlp = model.network['VisionMLP']

obs = torch.load('cohorts/ssv_BC_GT_CENTROID_V3/observation_data/InstinctJester_gt/flightroom_ssv_exp_clock/observations00050.pt', map_location='cpu', weights_only=False)
xnn = obs['data'][0]['Xnn'][60]

# History encoding
dxu = xnn['dxu_par'].unsqueeze(0)
_, zpar = hist_enc(dxu)

# VisionMLP encoding — forward returns (y_vis, None), y_vis is used as zimg
img_vis = xnn['img_vis'].unsqueeze(0)
tx_vis = xnn['tx_vis'].unsqueeze(0)
y_vis, _ = vis_mlp(img_vis, tx_vis)

# tx_com
tx_com = xnn['tx_com'].unsqueeze(0)

print(f"zpar shape: {zpar.shape}, y_vis shape: {y_vis.shape}, tx_com shape: {tx_com.shape}")

# Test sensitivity to obj_com
test_cases = [
    ('real centroid',      xnn['obj_com']),
    ('zero (occluded)',    torch.tensor([0.0, 0.0, 0.0])),
    ('opposite bearing',  torch.tensor([-0.5, 0.3, 1.0])),
    ('high elevation',    torch.tensor([0.04, 0.8, 1.0])),
    ('vis=0 but nonzero', torch.tensor([0.04, -0.71, 0.0])),
]

header = f"{'Case':<22s} | wz        vx       | max diff from real"
print(f"\n{header}")
print('-' * 70)
with torch.no_grad():
    real_out = None
    for name, oc in test_cases:
        out, _ = commander(tx_com, oc.unsqueeze(0), zpar, y_vis)
        o = out.squeeze().numpy()
        if real_out is None:
            real_out = o.copy()
            diff = 0.0
        else:
            diff = np.abs(o - real_out).max()
        print(f"{name:<22s} | {o[0]:+.5f}  {o[1]:+.5f}  | {diff:.5f}")
