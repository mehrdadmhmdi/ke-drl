import math

import torch

from ke_drl.rank_diagnostics import matrix_rank_diagnostics


def test_matrix_rank_diagnostics_detects_exact_rank():
    U = torch.eye(4, dtype=torch.float64)[:, :2]
    V = torch.eye(5, dtype=torch.float64)[:2, :]
    B = U @ torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64)) @ V

    diag = matrix_rank_diagnostics(B, prefix="B_")

    assert diag["B_num_rows"] == 4
    assert diag["B_num_cols"] == 5
    assert diag["B_numerical_rank"] == 2
    assert diag["B_rank_rel_1em2"] == 2
    assert math.isclose(diag["B_stable_rank"], (3.0**2 + 1.0**2) / 3.0**2)


def test_matrix_rank_diagnostics_effective_rank_is_one_for_rank_one():
    B = torch.ones(6, 3)

    diag = matrix_rank_diagnostics(B)

    assert diag["numerical_rank"] == 1
    assert math.isclose(diag["effective_rank"], 1.0, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(diag["stable_rank"], 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_matrix_rank_diagnostics_can_return_singular_values():
    B = torch.diag(torch.tensor([4.0, 2.0, 0.5]))

    diag = matrix_rank_diagnostics(B, return_singular_values=True)

    s = diag["singular_values"]
    assert isinstance(s, torch.Tensor)
    assert torch.allclose(s, torch.tensor([4.0, 2.0, 0.5]))
