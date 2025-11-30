# src/ke_drl/__init__.py
import os
os.environ.setdefault("VISPY_GL_BACKEND", "egl")
os.environ.setdefault("VISPY_USE_APP", "headless")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"
)

from .KE_DRL import KE_DRL
# ke_drl/__init__.py
from .api import (
    estimate_embedding,
    build_plot_config,
    plot_bellman_error,
    plot_total_loss,
    recover_joint_beta,
    compute_marginals_from_beta,
    plot_densities,
    mean_embedding_all,
    plot_operator_check_2d,
    save_weights_and_grid,
    cli,
    get_dataset,
    embedding_test_risk,
)

__all__ = [
    "estimate_embedding",
    "build_plot_config",
    "plot_bellman_error",
    "plot_total_loss",
    "recover_joint_beta",
    "compute_marginals_from_beta",
    "plot_densities",
    "mean_embedding_all",
    "plot_operator_check_2d",
    "save_weights_and_grid",
    "cli",
    "get_dataset", 
    "embedding_test_risk",
]


