# src/rk_drl/__init__.py
from .RK_DRL import RK_DRL
from .api import (
    run_RKDRL, build_plot_config,
    plot_bellman_error, plot_total_loss,
    recover_joint_beta, compute_marginals_from_beta, plot_densities,
    compute_L2_marginal_error, mean_embedding_all,
    plot_bland_altman, plot_quantile_calibration,
    plot_error_vs_distance_from_mode, plot_operator_check_2d,
    plot_error_heatmap, plot_statistics, save_weights_and_grid,
)

__all__ = [
    "RK_DRL", "run_RKDRL", "build_plot_config",
    "plot_bellman_error", "plot_total_loss",
    "recover_joint_beta", "compute_marginals_from_beta", "plot_densities",
    "compute_L2_marginal_error", "mean_embedding_all",
    "plot_bland_altman", "plot_quantile_calibration",
    "plot_error_vs_distance_from_mode", "plot_operator_check_2d",
    "plot_error_heatmap", "plot_statistics", "save_weights_and_grid",
]
