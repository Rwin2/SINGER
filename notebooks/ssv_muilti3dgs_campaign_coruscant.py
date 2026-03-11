import os
os.environ["ACADOS_SOURCE_DIR"] = "/data/erwinpi/FiGS-Standalone/acados"
os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH", "") + ":/data/erwinpi/FiGS-Standalone/acados/lib"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Disable Albumentations update check

import typer
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional
import wandb

from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import sousvide.synthesize.rollout_generator as rg
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_learning as pl
import sousvide.flight.deploy_ssv as df


app = typer.Typer()

# Monkey-patch Plotly.show to capture figures
_all_plotly_figs: List[go.Figure] = []
_original_show = go.Figure.show

def _capture_and_show(self, *args, **kwargs):
    _all_plotly_figs.append(self)
    return _original_show(self, *args, **kwargs)
go.Figure.show = _capture_and_show


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def init_wandb(cfg: dict, job: str) -> None:
    if not cfg.get("use_wandb"):
        return
    init_args = dict(
        project=cfg.get("wandb_project", "default_project"),
        name=cfg.get("wandb_run_name", job),
        config=cfg,
    )
    run_id = cfg.get("wandb_run_id")
    if run_id:
        init_args["id"] = run_id
        init_args["resume"] = cfg.get("wandb_resume", "allow")
    wandb.init(**init_args)


def common_options(
    config_file: Path,
    plot: bool,
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    wandb_run_id: Optional[str] = None,
    wandb_resume: Optional[str] = "allow",
) -> dict:
    cfg = load_yaml(config_file)
    cfg.update({
        "plot":           plot,
        "use_wandb":      use_wandb,
        "wandb_project":  wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_run_id":   wandb_run_id,
        "wandb_resume":   wandb_resume,
    })
    return cfg


def safe_to_image(fig, width=1200, height=1200, scale=1.0, min_size=200):
    """
    Try to render with Kaleido at the given size; if buffer-allocation fails,
    halve the dimensions (down to min_size) and retry. If even min_size fails,
    return None to signal "skip this image."
    """
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except ValueError as e:
        if "buffer allocation failed" in str(e):
            new_w, new_h = width // 2, height // 2
            if new_w < min_size or new_h < min_size:
                typer.echo(
                    f"Skipping image: smallest size {width}×{height} still too large",
                    err=True
                )
                return None
            typer.echo(
                f"Size {width}×{height} too big, retrying at {new_w}×{new_h}...",
                err=True
            )
            return safe_to_image(fig, new_w, new_h, scale, min_size)
        raise


def _log_figures_to_wandb(prefix: str) -> None:
    """Utilitaire : logue toutes les figures Matplotlib et Plotly en cours vers W&B."""
    logs = {}
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig_mpl = plt.figure(num)
        logs[f"{prefix}_mpl_fig_{i}"] = wandb.Image(fig_mpl)

    for i, fig in enumerate(_all_plotly_figs, start=1):
        img_bytes = safe_to_image(fig, width=1200, height=1200)
        if img_bytes is None:
            continue
        buf = BytesIO(img_bytes)
        pil_img = Image.open(buf)
        logs[f"{prefix}_plotly_png_{i}"] = wandb.Image(pil_img)

    if wandb.run is not None: wandb.log(logs)
    plt.close("all")
    _all_plotly_figs.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Commandes Typer
# ──────────────────────────────────────────────────────────────────────────────

@app.command()
def train_rl(config_file: str):
    import yaml
    from sousvide.instruct.train_policy_unified import train_rl_policy

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    cohort  = cfg["cohort"]
    method  = cfg["method"]
    flights = [tuple(x) for x in cfg["flights"]]
    roster  = cfg.get("roster") or ["InstinctJester"]

    train_rl_policy(
        cohort_name=cohort,
        roster=roster,
        method_name=method,
        flights=flights,
        Neps=50,
        train_on_failures_only=True,
        advantage_method="monte_carlo",
    )


@app.command("generate-rollouts")
def generate_rollouts(
    config_file: Path = typer.Option(..., exists=True),
    validation_mode: bool = typer.Option(False),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None, help="Existing W&B run ID to resume"),
    wandb_resume: Optional[str] = typer.Option("allow", help="resume mode: allow|must"),
):
    cfg = common_options(config_file, plot, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "generate_rollouts")
    rg.generate_rollout_data(
        cfg["cohort"], cfg["method"], cfg["flights"],
        validation_mode=validation_mode
    )

    if cfg.get("use_wandb"):
        _log_figures_to_wandb("generate_rollout")

    if cfg["plot"]:
        fig = ps.plot_rollout_data(cfg["cohort"])
        if cfg["use_wandb"]:
            if wandb.run is not None: wandb.log({"rollout_plot": fig})


@app.command("generate-observations")
def generate_observations(
    config_file: Path = typer.Option(..., exists=True),
    validation_mode: bool = typer.Option(False),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None, help="Existing W&B run ID to resume"),
    wandb_resume: Optional[str] = typer.Option("allow", help="resume mode: allow|must"),
):
    cfg = common_options(config_file, plot, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "generate_observations")
    og.generate_observation_data(
        cfg["cohort"], cfg["roster"],
        validation_mode=validation_mode
    )
    if cfg["plot"]:
        fig = ps.plot_observation_data(cfg["cohort"], cfg["roster"])
        if cfg["use_wandb"]:
            if wandb.run is not None: wandb.log({"observation_plot": fig})


@app.command("train-history")
def train_history(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(config_file, plot, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "train_history")
    tp.train_roster(
        cfg["cohort"], cfg["roster"], "Parameter",
        cfg["Nep_his"], lim_sv=cfg.get("lim_sv", 10)
    )
    if cfg["plot"]:
        fig = pl.plot_losses(cfg["cohort"], cfg["roster"], "Parameter")
        if cfg["use_wandb"]:
            if wandb.run is not None: wandb.log({"history_loss_plot": fig})


@app.command("train-command")
def train_command(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None, help="Existing W&B run ID to resume"),
    wandb_resume: Optional[str] = typer.Option("allow", help="resume mode: allow|must"),
):
    cfg = common_options(config_file, plot, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "train_command")
    tp.train_roster(
        cfg["cohort"], cfg["roster"], "Commander",
        cfg["Nep_com"], lim_sv=cfg.get("lim_sv", 10)
    )
    if cfg["plot"]:
        fig = pl.plot_losses(cfg["cohort"], cfg["roster"], "Commander")
        if cfg["use_wandb"]:
            if wandb.run is not None: wandb.log({"command_loss_plot": fig})


@app.command("dagger")
def train_dagger(
    config_file: Path = typer.Option(..., exists=True),
    n_iterations: int = typer.Option(10, help="Nombre d'itérations DAgger"),
    beta_start: float = typer.Option(0.7),
    beta_decay: float = typer.Option(0.85),
    collision_threshold: float = typer.Option(0.15),
    drift_threshold: float = typer.Option(2.0),
    expert_type: str = typer.Option("mpc", help="Expert type: mpc | potential | rrt"),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None),
    wandb_resume: Optional[str] = typer.Option("allow"),
):
    from sousvide.instruct.train_dagger import train_dagger_policy

    cfg = common_options(
        config_file, plot, use_wandb, wandb_project, wandb_run_name,
        wandb_run_id=wandb_run_id, wandb_resume=wandb_resume,
    )
    init_wandb(cfg, "train_dagger")

    typer.echo("=" * 70)
    typer.echo(f"[DAgger] Démarrage  —  {n_iterations} itérations")
    typer.echo(f"         β initial  : {beta_start}  |  decay : {beta_decay}")
    typer.echo(f"         collision  : {collision_threshold} m")
    typer.echo(f"         dérive max : {drift_threshold} m")
    typer.echo(f"         W&B        : {'ON  → ' + cfg.get('wandb_project','') if use_wandb else 'OFF'}")
    typer.echo("=" * 70)

    all_metrics = train_dagger_policy(
        cohort_name=cfg["cohort"],
        method_name=cfg["method"],
        roster=cfg.get("roster") or ["InstinctJester"],
        flights=[tuple(x) for x in cfg["flights"]],
        n_iterations=n_iterations,
        beta_start=beta_start,
        beta_decay=beta_decay,
        collision_threshold=collision_threshold,
        drift_threshold=drift_threshold,
        Nep_per_iter=cfg.get("Nep_dagger", 50),
        use_wandb=cfg.get("use_wandb", False),
        wandb_project=cfg.get("wandb_project", "singer-dagger"),
        wandb_run_name=cfg.get("wandb_run_name", "dagger"),
        lim_sv=cfg.get("lim_sv", 10),
        expert_type=expert_type,
    )

    # ── Résumé terminal ───────────────────────────────────────────────────────
    typer.echo("\n" + "=" * 70)
    typer.echo("[DAgger] RÉSUMÉ FINAL")
    typer.echo("=" * 70)
    for pilot_name, iter_metrics in all_metrics.items():
        typer.echo(f"\nPilot : {pilot_name}")
        typer.echo(f"  {'Iter':>4}  {'β':>6}  {'Collisions':>10}  {'Dérives':>8}  {'Succès':>8}")
        typer.echo(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")
        for m in iter_metrics:
            typer.echo(
                f"  {m['iteration']:>4}  {m['beta']:>6.3f}"
                f"  {m['collision_rate']:>10.1%}"
                f"  {m['drift_rate']:>8.1%}"
                f"  {m['success_rate']:>8.1%}"
            )

    if cfg.get("use_wandb"):
        if plot:
            _log_figures_to_wandb("dagger_summary")
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


@app.command()
def simulate(
    config_file: Path = typer.Option(..., exists=True),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None, help="Existing W&B run ID to resume"),
    wandb_resume: Optional[str] = typer.Option("allow", help="resume mode: allow|must"),
):
    cfg = common_options(config_file, False, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "simulate")
    df.simulate_roster(
        cfg["cohort"], cfg["method"], cfg["flights"], cfg["roster"],
        review=cfg["review"]
    )

    if cfg.get("use_wandb"):
        _log_figures_to_wandb("simulate")


@app.command()
def debug_trajectory(
    config_file: Path = typer.Option(..., exists=True),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    import glob
    import pickle
    import numpy as np
    import torch
    import figs.utilities.trajectory_helper as th
    import figs.visualize.plot_trajectories as pt

    cfg = common_options(config_file, False, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "debug_trajectory")

    workspace_path   = Path(__file__).resolve().parents[1]
    scenes_cfg_dir   = workspace_path / "configs" / "scenes"
    cohort_path_base = workspace_path / "cohorts" / cfg["cohort"]

    for scene_name, _ in cfg["flights"]:
        scene_cfg_file = scenes_cfg_dir / f"{scene_name}.yml"
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)

        combined_prefix = scenes_cfg_dir / scene_name
        for combined_path in glob.glob(f"{combined_prefix}*.pkl"):
            with open(combined_path, "rb") as f:
                data = pickle.load(f)

            base     = Path(combined_path).stem
            obj_name = base.replace(f"{scene_name}_", "")

            expert_filename = (
                cohort_path_base / f"sim_data_{scene_name}_{obj_name}_expert.pt"
            )
            if expert_filename.exists():
                expert_data = torch.load(expert_filename)
                typer.echo(f"expert_data length: {len(expert_data)}")
                try:
                    pt.plot_RO_time(
                        (expert_data[-1]["Tro"],
                         expert_data[-1]["Xro"],
                         expert_data[-1]["Uro"]),
                        plot_p=False, plot_q=True, aesthetics=False
                    )
                    pt.plot_RO_time(
                        (data["tXUi"][0],
                         data["tXUi"][1:11],
                         data["tXUi"][11:15, :-1]),
                        plot_p=False, plot_q=True, aesthetics=False
                    )
                except Exception:
                    typer.echo(
                        f"Error occurred. expert_data[-1] type: {type(expert_data[-1])}"
                    )
                    if isinstance(expert_data[-1], dict):
                        typer.echo(
                            f"expert_data[-1] keys: {list(expert_data[-1].keys())}"
                        )
                    raise ValueError("Error processing trajectories")
            else:
                th.debug_figures_RRT(
                    data["obj_loc"],
                    data["positions"],
                    data["trajectory"],
                    data["smooth_trajectory"],
                    data["times"],
                )

            def process_quaternions(data_array, label):
                ncols   = data_array.shape[1]
                indices = np.linspace(0, ncols - 1, num=10, dtype=int)
                for i in indices:
                    qx, qy, qz, qw = data_array[7:11, i]
                    t     = data_array[0, i]
                    roll  = np.arctan2(
                        2.0 * (qw * qx + qy * qz),
                        1.0 - 2.0 * (qx * qx + qy * qy)
                    )
                    pitch = np.arcsin(
                        np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
                    )
                    yaw   = np.arctan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz)
                    )
                    typer.echo(
                        f"{obj_name} t={t:.2f}: "
                        f"roll={roll:.4f} rad, "
                        f"pitch={pitch:.4f} rad, "
                        f"yaw={yaw:.4f} rad"
                    )

            process_quaternions(data["tXUi"],      "tXUi")
            process_quaternions(data["trajectory"], "trajectory")

    if cfg.get("use_wandb"):
        _log_figures_to_wandb("debug")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()