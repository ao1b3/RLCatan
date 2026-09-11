"""Load checkpoints saved before these modules moved into the rlcatan package."""
import importlib
import sys

# SB3 pickles the policy and feature-extractor classes by module path, so runs
# trained while these modules sat at the repository root refer to them by their
# bare names. Alias those names to the package submodules on demand.
LEGACY_MODULES = ("game", "features", "agents", "action_policy", "project_policy",
                  "multiplayer", "teacher", "strong_teacher", "training", "benchmark")


def install_legacy_module_aliases():
    for name in LEGACY_MODULES:
        sys.modules.setdefault(name, importlib.import_module("." + name, __package__))
