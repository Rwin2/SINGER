import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(7, 5))

start = np.array([0.5, 0.5])
goal = np.array([8.5, 6.5])

# Green trajectory: straight line
t = np.linspace(0, 1, 100)
green_x = start[0] + t * (goal[0] - start[0])
green_y = start[1] + t * (goal[1] - start[1])
ax.plot(green_x, green_y, color='#2ca02c', linewidth=3, zorder=3)

# Orange trajectory: sinuous path (perpendicular deviations)
t2 = np.linspace(0, 1, 300)
mid_x = start[0] + t2 * (goal[0] - start[0])
mid_y = start[1] + t2 * (goal[1] - start[1])
# Direction and perpendicular
dx, dy = goal - start
length = np.sqrt(dx**2 + dy**2)
perp_x, perp_y = -dy / length, dx / length
# Sinuous deviations perpendicular to the straight line
deviation = 1.4 * np.sin(3 * np.pi * t2) + 0.7 * np.sin(6.5 * np.pi * t2)
taper = np.sin(np.pi * t2) ** 1.2
offset = deviation * taper
orange_x = mid_x + offset * perp_x
orange_y = mid_y + offset * perp_y
ax.plot(orange_x, orange_y, color='#e67e22', linewidth=3, zorder=2)

# Start position: filled circle
ax.plot(*start, 'o', color='#2c3e50', markersize=16, zorder=5)
ax.text(start[0], start[1] - 0.55, 'Start', fontsize=13, ha='center',
        fontweight='bold', color='#2c3e50')

# Goal: gold star
ax.plot(*goal, marker='*', color='#f1c40f', markersize=30, zorder=5,
        markeredgecolor='#d4a017', markeredgewidth=1.2)
ax.text(goal[0], goal[1] - 0.55, 'Goal', fontsize=13, ha='center',
        fontweight='bold', color='#d4a017')

# Cross on orange trajectory (around midpoint)
cross_idx = 150
cx, cy = orange_x[cross_idx], orange_y[cross_idx]
ax.plot(cx, cy, 'x', color='#c0392b', markersize=18, markeredgewidth=3.5, zorder=6)

# Checkmark on green trajectory
check_idx = 50
gx, gy = green_x[check_idx], green_y[check_idx]
ax.text(gx + 0.15, gy + 0.45, u'\u2713', fontsize=22, color='#27ae60',
        fontweight='bold', zorder=6)

# Legend
green_patch = mpatches.Patch(color='#2ca02c', label='Preferred trajectory')
orange_patch = mpatches.Patch(color='#e67e22', label='Non-preferred trajectory')
ax.legend(handles=[green_patch, orange_patch], fontsize=12, loc='upper left',
          framealpha=0.9)

ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 8)
ax.set_aspect('equal')
ax.axis('off')

fig.tight_layout()
fig.savefig('/home/erwinpi/data/SINGER/docs/trajectory_preference.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved.")
