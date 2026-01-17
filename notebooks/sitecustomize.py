import sys, importlib, pkgutil

def _alias_controller():
    # 1) Point "controller" at your real "sousvide.control"
    real_ctrl = importlib.import_module("sousvide.control")
    sys.modules["controller"] = real_ctrl

    # 2) Make "controller.policies" point at sousvide.control.policies
    real_pols = importlib.import_module("sousvide.control.policies")
    sys.modules["controller.policies"] = real_pols

    # 3) Recursively alias all sub‐modules under sousvide.control.policies
    prefix = "sousvide.control.policies."
    for finder, full, ispkg in pkgutil.walk_packages(real_pols.__path__, prefix):
        mod = importlib.import_module(full)
        alias = full.replace("sousvide.control", "controller", 1)
        sys.modules[alias] = mod

_alias_controller()