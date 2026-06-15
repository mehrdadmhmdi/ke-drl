#!/usr/bin/env python3
"""
Downstream density-recovery diagnostic (NumPy only; no torch/scipy needed).

Re-recovers the total_clicks distribution from the ALREADY-SAVED mean-embedding
coefficients beta (density_atom_weights_table.csv) for each policy, comparing:
  saved     : the density_weight column from the pipeline run
  rkhs_proj : argmin_{w in simplex} (w-beta)^T K (w-beta)   [well-posed]
  pos_beta  : w proportional to max(beta, 0)

It also reports corr(beta_rev, beta_clk). After the eta fix, expect:
  - beta-corr clearly positive (the two policies' embeddings share structure)
  - frac_correct(sep>0) well above 0.5 and stable across recovery methods.

Pairs the two policies by their SHARED config directory, so it handles both the
old `main_cfg_N/` tree and the new `main_cfg_N_<target_point>/` tree.

Usage (scan everything under the current folder, i.e. the live run + archives):
    python3 downstream_recovery_diagnostic.py
Scan a specific tree / limit how many configs:
    RESULTS=evaluation_results/expedia_encoded_10k_linear3_fixed LIMIT=12 \
        python3 downstream_recovery_diagnostic.py
"""
import csv, glob, math, os, re
import numpy as np

# Root to scan recursively for density_atom_weights_table.csv files.
RESULTS = os.environ.get("RESULTS", ".")


def _matern_coeffs(p):
    c = [float(math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)))
         for m in range(p + 1)]
    return c, math.factorial(p) / math.factorial(2 * p)


def matern(X1, X2, nu, ell, sigma=1.0):
    """Half-integer Matern, matching ke_drl.matern_kernel."""
    p = int(round(nu - 0.5))
    x2 = (X1 ** 2).sum(1)[:, None]
    y2 = (X2 ** 2).sum(1)[None, :]
    d = np.sqrt(np.maximum(x2 + y2 - 2.0 * (X1 @ X2.T), 0.0)) / float(ell)
    a, pf = _matern_coeffs(p)
    pref = (sigma ** 2) * pf
    z = d * math.sqrt(2.0 * nu)
    t = 2.0 * z
    r = np.full_like(d, a[-1])
    for m in range(p - 1, -1, -1):
        r = r * t + a[m]
    return pref * np.exp(-z) * r


def proj_simplex(v):
    v = np.asarray(v, float).reshape(-1)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    th = css[rho - 1] / float(rho)
    w = np.maximum(v - th, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def _lmax(K, it=30):
    v = np.ones(K.shape[0]) / math.sqrt(K.shape[0])
    lam = 1.0
    for _ in range(it):
        u = K @ v
        lam = np.linalg.norm(u)
        if lam <= 0:
            break
        v = u / lam
    return float(lam)


def rkhs_proj(beta, K, max_iter=500, tol=1e-12):
    beta = np.asarray(beta, float).reshape(-1)
    K = 0.5 * (K + K.T) + 1e-8 * np.eye(K.shape[0])
    L = max(_lmax(K), 1e-8)
    w = proj_simplex(beta)
    y = w.copy()
    t = 1.0
    op = float((w - beta) @ K @ (w - beta))
    for _ in range(max_iter):
        g = K @ (y - beta)
        wn = proj_simplex(y - g / L)
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        y = wn + ((t - 1) / tn) * (wn - w)
        st = np.linalg.norm(wn - w)
        on = float((wn - beta) @ K @ (wn - beta))
        w, t = wn, tn
        if abs(op - on) <= tol * max(1.0, abs(op)) and st <= math.sqrt(tol):
            break
        op = on
    return proj_simplex(w)


def _load(path):
    with open(path) as fh:
        r = csv.reader(fh)
        H = next(r)
        rows = [x for x in r]
    return {h: i for i, h in enumerate(H)}, np.array(rows, float), H


def _kparse(dirname):
    """Kernel params from the per-policy folder name; fall back to 3.5 / 2.5."""
    mnu = re.search(r'nuZ([0-9.]+)', dirname)
    mell = re.search(r'ellZ([0-9.]+)', dirname)
    nu = float(mnu.group(1)) if mnu else 3.5
    ell = float(mell.group(1)) if mell else 2.5
    return nu, ell


def pmf_clicks(w, clk, kmax=8):
    supp = np.arange(0, kmax + 1)
    idx = np.clip(np.rint(np.maximum(clk, 0)).astype(int), 0, kmax)
    pmf = np.zeros(kmax + 1)
    for i, k in enumerate(idx):
        pmf[k] += w[i]
    s = pmf.sum()
    pmf = pmf / s if s > 0 else pmf
    return float((supp * pmf).sum()), pmf


def effn(w):
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def find_pairs():
    """Group rev/clk policies by their SHARED parent config directory."""
    pat = os.path.join(RESULTS, "**", "density_atom_weights_table.csv")
    g = {}
    for f in glob.glob(pat, recursive=True):
        policy_dir = os.path.basename(os.path.dirname(f))
        tag = ("clk" if "total_clicks" in policy_dir
               else ("rev" if "gross_revenue" in policy_dir else None))
        if tag is None:
            continue
        cfg_dir = os.path.dirname(os.path.dirname(f))  # the main_cfg_..._<target> dir
        g.setdefault(cfg_dir, {})[tag] = f
    return {k: v for k, v in g.items() if "rev" in v and "clk" in v}


def recover(path):
    idx, arr, H = _load(path)
    beta = arr[:, idx["beta_mean_embedding"]]
    wsav = arr[:, idx["density_weight"]]
    zn = [c for c in H if c.startswith("Z_norm_")]
    craw = [c for c in H if c.startswith("Z_raw_") and "click" in c.lower()][0]
    Zn = np.stack([arr[:, idx[c]] for c in zn], 1)
    clk = arr[:, idx[craw]]
    nu, ell = _kparse(os.path.basename(os.path.dirname(path)))
    K = matern(Zn, Zn, nu, ell, 1.0)
    pb = np.maximum(beta, 0.0)
    return {
        "saved": wsav / wsav.sum(),
        "rkhs_proj": rkhs_proj(beta, K),
        "pos_beta": pb / pb.sum() if pb.sum() > 0 else np.full_like(beta, 1.0 / beta.size),
    }, clk, beta


def _label(cfg_dir):
    base = os.path.basename(cfg_dir)
    return base[-26:] if len(base) > 26 else base


def main():
    P = find_pairs()
    lim = int(os.environ.get("LIMIT", "0"))
    items = sorted(P.items(), key=lambda x: x[0])
    if lim > 0:
        items = items[:lim]
    print("RESULTS=%s" % os.path.abspath(RESULTS))
    print("Found %d rev/clk config pairs (running %d)\n" % (len(P), len(items)), flush=True)
    print("%-28s %-10s %9s %9s %7s %8s %8s" %
          ("config", "variant", "E[clk]rev", "E[clk]clk", "sep", "effNrev", "effNclk"))
    variants = ["saved", "rkhs_proj", "pos_beta"]
    agg = {v: [] for v in variants}
    betacorr = []
    for cfg_dir, d in items:
        rev, crev, brev = recover(d["rev"])
        clk, cclk, bclk = recover(d["clk"])
        if brev.shape == bclk.shape and np.std(brev) > 0 and np.std(bclk) > 0:
            betacorr.append(float(np.corrcoef(brev, bclk)[0, 1]))
        lab = _label(cfg_dir)
        for v in variants:
            Er, _ = pmf_clicks(rev[v], crev)
            Ec, _ = pmf_clicks(clk[v], cclk)
            sep = Ec - Er
            agg[v].append(sep)
            print("%-28s %-10s %9.3f %9.3f %+7.3f %8.1f %8.1f" %
                  (lab, v, Er, Ec, sep, effn(rev[v]), effn(clk[v])), flush=True)
        print(flush=True)
    print("=" * 72)
    print("sep = E[clk]_clk - E[clk]_rev   (want > 0 : click policy => more clicks)")
    print("=" * 72)
    for v in variants:
        s = np.array(agg[v])
        if s.size:
            print("  %-10s mean=%+.3f median=%+.3f frac_correct(>0)=%.2f n=%d" %
                  (v, s.mean(), np.median(s), np.mean(s > 0), s.size))
    if betacorr:
        bc = np.array(betacorr)
        print("\n  beta(rev) vs beta(clk) correlation: median=%.3f  (near 0 => policy signal is noise)"
              % np.median(bc))


if __name__ == "__main__":
    main()
