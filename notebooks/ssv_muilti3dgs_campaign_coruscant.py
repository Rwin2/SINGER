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


@app.command("rl-finetune")
def rl_finetune(
    config_file: Path = typer.Option(..., exists=True),
):
    """Step 7: RL fine-tuning of a pre-trained DAgger/BC model using off-policy actor-critic (UTD=5)."""
    import yaml
    from sousvide.instruct.train_rl_finetune import train_rl_finetune

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    flights = [tuple(x) for x in cfg["flights"]]
    roster = cfg.get("roster") or ["InstinctJester"]

    train_rl_finetune(
        cohort_name=cfg["cohort"],
        method_name=cfg["method"],
        roster=roster,
        flights=flights,
        source_cohort=cfg.get("source_cohort"),
        bc_cohort=cfg.get("bc_cohort"),
        # RL hyperparameters
        n_iterations=cfg.get("n_iterations", 20),
        rollouts_per_iter=cfg.get("rollouts_per_iter", 6),
        utd=cfg.get("utd", 5),
        batch_size=cfg.get("batch_size", 128),
        critic_lr=cfg.get("critic_lr", 3e-4),
        actor_lr=cfg.get("actor_lr", 1e-5),
        critic_target_tau=cfg.get("critic_target_tau", 0.005),
        discount=cfg.get("discount", 0.99),
        noise_std=cfg.get("noise_std", 0.05),
        stddev_clip=cfg.get("stddev_clip", 0.3),
        num_critics=cfg.get("num_critics", 2),
        critic_hidden=cfg.get("critic_hidden", 256),
        replay_capacity=cfg.get("replay_capacity", 200000),
        warmup_transitions=cfg.get("warmup_transitions", 500),
        # Reward weights
        goal_progress_weight=cfg.get("goal_progress_weight", 1.0),
        collision_penalty_weight=cfg.get("collision_penalty_weight", 2.0),
        clearance_threshold=cfg.get("clearance_threshold", 0.5),
        fov_bonus=cfg.get("fov_bonus", 0.05),
        success_bonus=cfg.get("success_bonus", 10.0),
        collision_penalty=cfg.get("collision_penalty", 10.0),
        time_penalty=cfg.get("time_penalty", 0.01),
        # BC regularization
        bc_reg_freq=cfg.get("bc_reg_freq", 1),
        bc_reg_weight=cfg.get("bc_reg_weight", 1.0),
        # Stability
        actor_grad_clip=cfg.get("actor_grad_clip", 0.5),
        critic_warmup_iters=cfg.get("critic_warmup_iters", 2),
        reset_to_best=cfg.get("reset_to_best", True),
        # Evaluation
        n_eval=cfg.get("n_eval", 20),
        eval_seed=cfg.get("eval_seed", 42),
        patience=cfg.get("patience", 5),
        # Resume
        resume_warmup=cfg.get("resume_warmup", True),
    )


@app.command("rl-ppo")
def rl_ppo(
    config_file: Path = typer.Option(..., exists=True),
):
    """Step 7b: RL fine-tuning via PPO with KL constraint to DAgger reference."""
    import yaml
    from sousvide.instruct.train_rl_ppo import train_rl_ppo

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    flights = [tuple(x) for x in cfg["flights"]]
    roster = cfg.get("roster") or ["InstinctJester"]

    train_rl_ppo(
        cohort_name=cfg["cohort"],
        method_name=cfg["method"],
        roster=roster,
        flights=flights,
        source_cohort=cfg.get("source_cohort"),
        bc_cohort=cfg.get("bc_cohort"),
        # PPO hyperparameters
        n_iterations=cfg.get("n_iterations", 30),
        rollouts_per_iter=cfg.get("rollouts_per_iter", 8),
        ppo_epochs=cfg.get("ppo_epochs", 3),
        batch_size=cfg.get("batch_size", 128),
        actor_lr=cfg.get("actor_lr", 1e-6),
        value_lr=cfg.get("value_lr", 3e-4),
        clip_eps=cfg.get("clip_eps", 0.1),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        value_coef=cfg.get("value_coef", 0.5),
        entropy_coef=cfg.get("entropy_coef", 0.01),
        kl_coef=cfg.get("kl_coef", 0.1),
        log_std_init=cfg.get("log_std_init", -2.0),
        value_hidden=cfg.get("value_hidden", 256),
        # Reward weights
        goal_progress_weight=cfg.get("goal_progress_weight", 1.0),
        collision_penalty_weight=cfg.get("collision_penalty_weight", 2.0),
        clearance_threshold=cfg.get("clearance_threshold", 0.5),
        fov_bonus=cfg.get("fov_bonus", 0.05),
        success_bonus=cfg.get("success_bonus", 10.0),
        collision_penalty=cfg.get("collision_penalty", 10.0),
        time_penalty=cfg.get("time_penalty", 0.01),
        # Stability
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        reset_to_best=cfg.get("reset_to_best", True),
        # Evaluation
        n_eval=cfg.get("n_eval", 20),
        eval_seed=cfg.get("eval_seed", 42),
        patience=cfg.get("patience", 8),
        use_unseen_eval=cfg.get("use_unseen_eval", False),
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
    n_rollouts_per_object: int = typer.Option(5, help="Number of different branches to fly per object per DAgger iteration"),
    max_trajectories: int = typer.Option(10, help="Number of benchmark trajectories per evaluation"),
    aggregate_dagger: bool = typer.Option(False, help="Cumulate all past DAgger data (True) or train only on current iter (False=online)"),
    start_pos_noise: float = typer.Option(0.5, help="Random position noise (m) added to initial state for trajectory diversity"),
    deviation_filter_dist: float = typer.Option(0.3, help="Keep annotations where drone drifted >this (m) from reference trajectory"),
    close_approach_dist: float = typer.Option(5.0, help="Always keep annotations within this distance (m) of goal"),
    run_simulate: bool = typer.Option(False, help="Run simulation + video generation after DAgger completes"),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(True, help="Enable W&B logging (default: True)"),
    wandb_project: Optional[str] = typer.Option("singer-dagger"),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_run_id: Optional[str] = typer.Option(None),
    wandb_resume: Optional[str] = typer.Option("allow"),
):
    from sousvide.instruct.train_dagger import train_dagger_policy
    from datetime import datetime

    # Auto-generate wandb run name from cohort if not specified
    if wandb_run_name is None:
        _cfg_tmp = load_yaml(config_file)
        wandb_run_name = f"dagger_{_cfg_tmp['cohort']}_{datetime.now().strftime('%m%d_%H%M')}"

    cfg = common_options(
        config_file, plot, use_wandb, wandb_project, wandb_run_name,
        wandb_run_id=wandb_run_id, wandb_resume=wandb_resume,
    )
    init_wandb(cfg, "train_dagger")

    # Define separate metric steps so training and DAgger don't conflict
    if use_wandb:
        try:
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")
            wandb.define_metric("dagger/*")
            wandb.define_metric("benchmark/*")
        except Exception:
            pass

    _n_iter = cfg.get("n_iterations", n_iterations)
    _beta_s = cfg.get("beta_start", beta_start)
    _beta_d = cfg.get("beta_decay", beta_decay)
    _pat    = cfg.get("patience", 2)
    typer.echo("=" * 70)
    typer.echo(f"[DAgger] Démarrage  —  {_n_iter} itérations  (patience={_pat})")
    typer.echo(f"         β initial  : {_beta_s}  |  decay : {_beta_d}")
    typer.echo(f"         collision  : {collision_threshold} m")
    typer.echo(f"         dérive max : {drift_threshold} m")
    typer.echo(f"         W&B        : {'ON  → ' + cfg.get('wandb_project','') if use_wandb else 'OFF'}")
    typer.echo("=" * 70)

    all_metrics = train_dagger_policy(
        cohort_name=cfg["cohort"],
        method_name=cfg["method"],
        roster=cfg.get("roster") or ["InstinctJester"],
        flights=[tuple(x) for x in cfg["flights"]],
        n_iterations=cfg.get("n_iterations", n_iterations),
        beta_start=cfg.get("beta_start", beta_start),
        beta_decay=cfg.get("beta_decay", beta_decay),
        collision_threshold=collision_threshold,
        drift_threshold=drift_threshold,
        Nep_per_iter=cfg.get("Nep_dagger", 50),
        use_wandb=cfg.get("use_wandb", False),
        wandb_project=cfg.get("wandb_project", "singer-dagger"),
        wandb_run_name=cfg.get("wandb_run_name", "dagger"),
        lim_sv=cfg.get("lim_sv", 10),
        max_trajectories=cfg.get("n_benchmark", max_trajectories),
        n_eval_per_iter=cfg.get("n_eval_per_iter", 10),
        expert_type=cfg.get("expert_type", expert_type),
        aggregate_dagger=cfg.get("aggregate_dagger", aggregate_dagger),
        start_pos_noise=cfg.get("start_pos_noise", start_pos_noise),
        n_rollouts_per_object=cfg.get("n_rollouts_per_object", n_rollouts_per_object),
        deviation_filter_dist=cfg.get("deviation_filter_dist", deviation_filter_dist),
        close_approach_dist=cfg.get("close_approach_dist", close_approach_dist),
        max_annotation_goal_dist=float(cfg.get("max_annotation_goal_dist", 50.0)),
        max_deviation_dist=float(cfg.get("max_deviation_dist", float('inf'))),
        dagger_lr=float(cfg.get("dagger_lr", 1e-5)),
        bc_cohort_name=cfg.get("bc_cohort", None),
        eval_seed=cfg.get("eval_seed", None),
        reset_to_best=cfg.get("reset_to_best", False),
        patience=cfg.get("patience", 2),
        dagger_only=cfg.get("dagger_only", False),
        dagger_oversample=int(cfg.get("dagger_oversample", 1)),
        orientation_deviation_deg=cfg.get("orientation_deviation_deg", None),
        max_orientation_dev_deg=float(cfg.get("max_orientation_dev_deg", 180.0)),
        # V10 enhancements (backward compatible)
        ewc_lambda=float(cfg.get("ewc_lambda", 0.0)),
        lr_schedule=cfg.get("lr_schedule", None),
        lr_decay_per_iter=float(cfg.get("lr_decay_per_iter", 1.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        # Collision-weighted loss
        collision_weight_alpha=float(cfg.get("collision_weight_alpha", 0.0)),
        collision_weight_threshold=float(cfg.get("collision_weight_threshold", 0.5)),
        # Data ratio control
        max_dagger_samples=int(cfg.get("max_dagger_samples", 0)),
    )

    # ── Résumé terminal ───────────────────────────────────────────────────────
    typer.echo("\n" + "=" * 70)
    typer.echo("[DAgger] RÉSUMÉ FINAL")
    typer.echo("=" * 70)
    for pilot_name, iter_metrics in all_metrics.items():
        typer.echo(f"\nPilot : {pilot_name}")
        typer.echo(f"  {'Iter':>4}  {'β':>6}  {'Collisions':>10}  {'FT_Success':>10}  {'Succès':>8}")
        typer.echo(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
        for m in iter_metrics:
            typer.echo(
                f"  {m['iteration']:>4}  {m['beta']:>6.3f}"
                f"  {m['collision_rate']:>10.1%}"
                f"  {m.get('full_traj_success', float('nan')):>10.1%}"
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

    # Auto-run simulation + video generation if requested
    if run_simulate:
        typer.echo("\n" + "=" * 70)
        typer.echo("[DAgger] Running simulation to generate videos...")
        typer.echo("=" * 70)
        df.simulate_roster(
            cfg["cohort"], cfg["method"], cfg["flights"], cfg["roster"],
            review=False,
        )
        typer.echo("[DAgger] Simulation + video generation complete!")


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
def benchmark(
    config_file: Path = typer.Option(..., exists=True),
    models: str = typer.Option(..., help="Comma-separated 'Label:cohort/pilot' specs"),
    branches: str = typer.Option("seen", help="seen | unseen | both"),
    max_trajectories: int = typer.Option(50, help="Runs per object per model"),
    seed: int = typer.Option(42, help="Benchmark seed"),
    seeds: Optional[str] = typer.Option(None, help="Comma-separated seeds for multi-seed mode"),
    save_plots: bool = typer.Option(True, help="Save plotly HTMLs"),
    save_videos: bool = typer.Option(True, help="Save MP4 videos"),
    save_analysis: bool = typer.Option(True, help="Save JSON results"),
    include_expert: bool = typer.Option(False, help="Include MPC expert"),
    overlay: bool = typer.Option(True, help="Multi-model overlay plotly"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
):
    """Unified benchmark: evaluate one or more models on seen/unseen branches."""
    from sousvide.instruct.benchmark import run_unified_benchmark, _parse_model_specs

    cfg = common_options(config_file, False, False, None, None)
    workspace_path = str(Path(__file__).resolve().parents[1])
    scenes_cfg_dir = str(Path(workspace_path) / "configs" / "scenes")
    bc_cohort = cfg.get("bc_cohort", cfg["cohort"])

    model_specs = _parse_model_specs(models)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else [seed]

    run_unified_benchmark(
        flights=cfg["flights"],
        scenes_cfg_dir=scenes_cfg_dir,
        model_specs=model_specs,
        bc_cohort=bc_cohort,
        branches_mode=branches,
        max_trajectories=max_trajectories,
        seeds=seed_list,
        save_plots=save_plots,
        save_videos=save_videos,
        save_analysis=save_analysis,
        include_expert=include_expert,
        overlay=overlay,
        output_dir=output_dir,
    )


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


@app.command("cross-benchmark")
def cross_benchmark(
    config_file: Path = typer.Option(..., exists=True, help="Config for scene/flight setup (e.g. ssv_dagger_potential.yml)"),
    cohort_before:    str = typer.Option("ssv_CLIPSEG_NORMAL",  help="Cohort with BC-only model"),
    cohort_potential: str = typer.Option("ssv_dagger_potential", help="Cohort after DAgger-potential"),
    cohort_rrt:       str = typer.Option("ssv_dagger_rrt",       help="Cohort after DAgger-RRT"),
    pilot_name: str        = typer.Option("InstinctJester"),
    benchmark_seed: int    = typer.Option(123, help="Seed for start-position sampling (use != DAgger seed 42)"),
    max_trajectories: int  = typer.Option(50,  help="Trajectories per object per model"),
    output: Optional[str]  = typer.Option(None, help="Path to write JSON results"),
):
    """
    Compare all 3 InstinctJester variants (before DAgger, after potential-field
    DAgger, after RRT DAgger) on the SAME held-out start conditions.

    Each model is evaluated on max_trajectories trajectories per object,
    sampled from the second half of tXUi (unseen during BC training) using
    the given benchmark_seed so conditions are identical across models.
    """
    from sousvide.instruct.train_dagger import run_cross_cohort_benchmark

    cfg = common_options(config_file, False, False, None, None)
    workspace_path = Path(__file__).resolve().parents[1]
    scenes_cfg_dir = str(workspace_path / "configs" / "scenes")
    flights        = [tuple(x) for x in cfg["flights"]]
    cohort_base    = str(workspace_path / "cohorts")

    def _model_path(cohort: str, label: str) -> str:
        # After DAgger the final checkpoint is saved as model_after_dagger.pth;
        # the original BC model is model.pth (the best-validation checkpoint).
        bench_path = (
            f"{cohort_base}/{cohort}/dagger_data/{pilot_name}/benchmark/model_after_dagger.pth"
        )
        if label == "before_dagger":
            bc_path = f"{cohort_base}/{cohort}/roster/{pilot_name}/model.pth"
            return bc_path
        return bench_path

    models = [
        {
            "label":      "before_dagger",
            "cohort":     cohort_before,
            "pilot_name": pilot_name,
            "model_path": _model_path(cohort_before, "before_dagger"),
        },
        {
            "label":      "after_potential",
            "cohort":     cohort_potential,
            "pilot_name": pilot_name,
            "model_path": _model_path(cohort_potential, "after_potential"),
        },
        {
            "label":      "after_rrt",
            "cohort":     cohort_rrt,
            "pilot_name": pilot_name,
            "model_path": _model_path(cohort_rrt, "after_rrt"),
        },
    ]

    typer.echo("=" * 70)
    typer.echo("[CrossBenchmark] Models:")
    for m in models:
        typer.echo(f"  {m['label']:20s}  {m['model_path']}")
    typer.echo(f"  seed={benchmark_seed}  n={max_trajectories}/obj")
    typer.echo("=" * 70 + "\n")

    out_path = output or str(
        workspace_path / "logs" / f"cross_benchmark_seed{benchmark_seed}.json"
    )

    run_cross_cohort_benchmark(
        models=models,
        flights=flights,
        scenes_cfg_dir=scenes_cfg_dir,
        benchmark_seed=benchmark_seed,
        max_trajectories=max_trajectories,
        output_path=out_path,
    )


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()