"""Empirical calibration check for PartitionScheduler vs NoiseGate.

For each sigma in a sweep:
  - Draw random Δ ~ N(0, I) of dim d.
  - Compute z = encode(Δ) in the partition basis.
  - Get (N, k) from the scheduler.
  - Compute block norms ||P_i Δ||² and positive-exponent softmax weights w_i.
  - Realized magnitude: r(σ) = ||Σ_i w_i P_i Δ||² / ||Δ||²
  - Compare mean ± std of r(σ) against noise_gate(σ).

Useful for deciding the empirical correction to apply to k(σ).

Run from the repo root:
    python diagnostics/partition_calibration.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from ppg.ppg import (
    NoiseGate, OrthogonalPartition, UniformRankScheduler,
    AmbientLatentMap, MatrixLatentMap,
)


def make_basis(kind, d, seed=0):
    if kind == "ambient":
        return AmbientLatentMap()
    if kind == "orthogonal":
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(d, d, generator=g)
        Q, _ = torch.linalg.qr(raw)
        return MatrixLatentMap(Q.T, mat_inv=Q)
    raise ValueError(kind)


@torch.no_grad()
def realized_magnitude(partition, sigma, n_samples, d, device="cpu"):
    """Return per-sample r = ||Σ_i w_i P_i Δ||² / ||Δ||² for n_samples random Δ."""
    delta = torch.randn(n_samples, d, device=device)
    z = partition.encode(delta)                        # (B, d)
    N, k = partition(sigma)
    if k == 0:
        return np.zeros(n_samples, dtype=np.float32), (N, k)
    active = N * k

    z_active = z[..., :active]                         # (B, N*k)
    z_blocks = z_active.view(n_samples, N, k)          # (B, N, k)

    ell = z_blocks.pow(2).sum(dim=-1)                  # (B, N) = ||P_i Δ||²
    w = torch.softmax(ell, dim=-1)                     # positive exponent

    z_out_active = (w.unsqueeze(-1) * z_blocks).reshape(n_samples, active)
    z_out = torch.zeros_like(z)
    z_out[..., :active] = z_out_active

    out = partition.decode(z_out)                      # (B, d)

    num = out.pow(2).sum(dim=-1)
    den = delta.pow(2).sum(dim=-1)
    return (num / den).cpu().numpy(), (N, k)


@torch.no_grad()
def realized_magnitude_flat(partition, sigma, n_samples, d, device="cpu"):
    """Same but with FLAT attention (w_i = 1/N) — the calibration baseline.
    Should yield E[r] = k/d exactly."""
    delta = torch.randn(n_samples, d, device=device)
    z = partition.encode(delta)
    N, k = partition(sigma)
    if k == 0:
        return np.zeros(n_samples, dtype=np.float32), (N, k)
    active = N * k

    z_active = z[..., :active]
    z_blocks = z_active.view(n_samples, N, k)
    w = torch.full((n_samples, N), 1.0 / N, device=device)

    z_out_active = (w.unsqueeze(-1) * z_blocks).reshape(n_samples, active)
    z_out = torch.zeros_like(z)
    z_out[..., :active] = z_out_active
    out = partition.decode(z_out)

    num = out.pow(2).sum(dim=-1)
    den = delta.pow(2).sum(dim=-1)
    return (num / den).cpu().numpy(), (N, k)


def run(args):
    device = args.device
    gate = NoiseGate(type_gate=args.gate, nu=args.nu, n=args.hill_n)
    gate = NoiseGate(type_gate=args.gate, nu=args.nu, n=args.hill_n)
    sched = UniformRankScheduler(gate, d=args.d, k_min=args.k_min)
    basis = make_basis(args.basis, args.d, seed=args.seed)
    partition = OrthogonalPartition(basis, sched).to(device)

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)

    gate_vals = []
    softmax_mean, softmax_std = [], []
    flat_mean, flat_std = [], []
    k_over_d = []
    Ns, ks = [], []

    torch.manual_seed(args.seed)
    for s in sigmas:
        sig_t = torch.tensor(float(s), device=device)
        gate_vals.append(float(gate(sig_t).item()))

        r, (N, k) = realized_magnitude(partition, sig_t, args.n_samples, args.d, device)
        softmax_mean.append(r.mean())
        softmax_std.append(r.std())

        rf, _ = realized_magnitude_flat(partition, sig_t, args.n_samples, args.d, device)
        flat_mean.append(rf.mean())
        flat_std.append(rf.std())

        k_over_d.append(k / args.d)
        Ns.append(N)
        ks.append(k)

    sigmas = np.asarray(sigmas)
    gate_vals = np.asarray(gate_vals)
    softmax_mean = np.asarray(softmax_mean)
    softmax_std = np.asarray(softmax_std)
    flat_mean = np.asarray(flat_mean)
    flat_std = np.asarray(flat_std)
    k_over_d = np.asarray(k_over_d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(sigmas, gate_vals, label="noise_gate(σ)", color="black", lw=2)
    ax.plot(sigmas, k_over_d, label="k(σ)/d (scheduler)", color="tab:gray", ls="--")
    ax.fill_between(sigmas, softmax_mean - softmax_std, softmax_mean + softmax_std,
                    alpha=0.25, color="tab:blue")
    ax.plot(sigmas, softmax_mean, label="realized (positive softmax)", color="tab:blue")
    ax.fill_between(sigmas, flat_mean - flat_std, flat_mean + flat_std,
                    alpha=0.2, color="tab:orange")
    ax.plot(sigmas, flat_mean, label="realized (flat attention)", color="tab:orange", ls=":")
    ax.set_xlabel("σ")
    ax.set_ylabel("E[ ||Σ wᵢ Pᵢ Δ||² / ||Δ||² ]")
    ax.set_title(
        f"basis={args.basis}  d={args.d}  gate={args.gate}(ν={args.nu})  "
        f"samples={args.n_samples}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bias = softmax_mean - gate_vals
    ax.plot(sigmas, bias, label="softmax − gate", color="tab:blue")
    ax.plot(sigmas, flat_mean - gate_vals, label="flat − gate", color="tab:orange", ls=":")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("σ")
    ax.set_ylabel("realized − noise_gate(σ)")
    ax.set_title("Calibration bias")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = args.out
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")

    if args.print_table:
        print(f"{'σ':>8} {'gate':>8} {'k/d':>8} {'mean':>10} {'std':>10} {'flat':>10} N  k")
        for s, g, kd, m, sd, fm, n_, k_ in zip(
            sigmas, gate_vals, k_over_d, softmax_mean, softmax_std, flat_mean, Ns, ks
        ):
            print(f"{s:8.3f} {g:8.4f} {kd:8.4f} {m:10.4f} {sd:10.4f} {fm:10.4f} {n_:>3} {k_:>3}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--n-sigma", type=int, default=40)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--gate", type=str, default="hill", choices=["quadratic", "hill", "heaviside"])
    p.add_argument("--nu", type=float, default=2.0)
    p.add_argument("--hill-n", type=int, default=3)
    p.add_argument("--k-min", type=int, default=0)
    p.add_argument("--basis", type=str, default="ambient", choices=["ambient", "orthogonal"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="report/partition_calibration.png")
    p.add_argument("--print-table", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run(args)
