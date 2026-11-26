from __future__ import annotations
import math
import time
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from triton_kmeans import (
    run_engine_v11_triton,
    triton_reproject_lloyd_step,
    triton_reproject_engine_step,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


# =====================================================================
# 3. CuBLAS Lloyd & e-means Baselines
# =====================================================================

@torch.no_grad()
def run_lloyd_cublas(X: torch.Tensor, k: int, iters: int = 10, init_C: torch.Tensor | None = None) -> torch.Tensor:
    n, d = X.shape
    C = init_C.clone() if init_C is not None else X[torch.randint(0, n, (k,), device=device)].clone()
    chunk_size = 65536

    for _ in range(iters):
        c_half = 0.5 * (C * C).sum(dim=1)
        new_C = torch.zeros_like(C)
        counts = torch.zeros(k, 1, device=X.device, dtype=X.dtype)

        for i in range(0, n, chunk_size):
            X_chunk = X[i : i + chunk_size]
            scores = torch.addmm(-c_half, X_chunk, C.T)
            assign = torch.argmax(scores, dim=1)
            O = torch.zeros(X_chunk.shape[0], k, device=X.device, dtype=X.dtype)
            O.scatter_(1, assign.unsqueeze(1), 1.0)
            new_C += O.T @ X_chunk
            counts += O.sum(dim=0, keepdim=True).T

        dead = counts == 0
        C = torch.where(dead, C, new_C / torch.clamp(counts, min=1.0))
    return C


def _emeans_step_cublas(B_X: torch.Tensor, C: torch.Tensor, S: torch.Tensor, N_stat: torch.Tensor, alpha: float):
    c_half_norm = 0.5 * (C * C).sum(dim=1)
    scores = torch.addmm(-c_half_norm, B_X, C.T)
    assign = torch.argmax(scores, dim=1)

    k = C.shape[0]
    O = torch.zeros(B_X.shape[0], k, device=B_X.device, dtype=torch.float32)
    O.scatter_(1, assign.unsqueeze(1), 1.0)

    s_batch = O.T @ B_X.to(torch.float32)
    n_batch = O.sum(dim=0, keepdim=True).T

    S = (1.0 - alpha) * S + alpha * s_batch
    N_stat = (1.0 - alpha) * N_stat + alpha * n_batch

    valid = N_stat.squeeze(1) > 1e-6
    new_C = (S / torch.clamp(N_stat, min=1e-6)).to(C.dtype)
    return torch.where(valid.unsqueeze(1), new_C, C), S, N_stat


@torch.no_grad()
def run_original_emeans(X: torch.Tensor, k: int, epochs: float = 1.0, init_C: torch.Tensor | None = None) -> torch.Tensor:
    n, d = X.shape
    B = 0.035 * (k ** 0.13) * (n ** 0.88) * (epochs ** 0.81)
    B = max(k * 2, min(int(round(B)), n))
    B = min(((B + 127) // 128) * 128, n)

    a0 = min(max(float(0.824 * (k ** -0.46) * (B ** 0.90) * (n ** -0.45) * (epochs ** -0.11)), 0.01), 1.0)
    n_batches = max(1, int((n * epochs) // B))

    C = init_C.clone() if init_C is not None else X[torch.randperm(n, device=device)[:k]].clone()
    N_stat = torch.full((k, 1), 1e-10, device=device, dtype=torch.float32)
    S = C.to(torch.float32) * 1e-10

    fac = torch.ones(d, device=device, dtype=torch.float32)
    fac[::2] += 1.0 / 1024.0
    fac[1::2] -= 1.0 / 1024.0

    for step in range(n_batches):
        progress = float(step) / float(n_batches)
        alpha = a0 * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))
        B_X = X[torch.randint(0, n, (B,), device=device)]
        C, S, N_stat = _emeans_step_cublas(B_X, C, S, N_stat, alpha)

        if (step + 1) % 5 == 0:
            dead_mask = N_stat.squeeze(1) < 0.1 * N_stat.mean().item()
            dead_indices = torch.where(dead_mask)[0]
            if len(dead_indices) > 0:
                active_weights = torch.clamp(N_stat.squeeze(1), min=0.0).clone()
                active_weights[dead_mask] = 0.0
                num_active = int((active_weights > 0).sum().item())
                if num_active > 0:
                    num_revive = min(len(dead_indices), num_active)
                    donors = torch.multinomial(active_weights, num_revive, replacement=False)
                    target_dead = dead_indices[:num_revive]
                    half_n = 0.5 * N_stat[donors]
                    c_donor = C[donors].to(torch.float32)
                    c1, c2 = c_donor * fac, c_donor * (2.0 - fac)
                    N_stat[donors], N_stat[target_dead] = half_n, half_n
                    S[donors], S[target_dead] = c1 * half_n, c2 * half_n
                    C[donors], C[target_dead] = c1.to(X.dtype), c2.to(X.dtype)

    return run_lloyd_cublas(X, k, iters=1, init_C=C)


# =====================================================================
# 4. Hungarian & Greedy Unscramblers
# =====================================================================

@torch.no_grad()
def unscramble_hungarian_cosine(C_old: torch.Tensor, C_new: torch.Tensor) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    sim = (F.normalize(C_old.float(), dim=-1) @ F.normalize(C_new.float(), dim=-1).T).cpu().numpy()
    _, col_ind = linear_sum_assignment(-sim)
    cost_ms = (time.perf_counter() - t0) * 1000.0
    perm = torch.tensor(col_ind, device=C_new.device, dtype=torch.long)
    return C_new[perm], cost_ms


@torch.no_grad()
def unscramble_greedy_bipartite(C_old: torch.Tensor, C_new: torch.Tensor) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    M = C_old.shape[0]
    sim = (F.normalize(C_old.float(), dim=-1) @ F.normalize(C_new.float(), dim=-1).T).cpu().tolist()
    pairs = sorted(((sim[i][j], i, j) for i in range(M) for j in range(M)), reverse=True)
    used_r, used_c = set(), set()
    perm = [0] * M
    for _, i, j in pairs:
        if i not in used_r and j not in used_c:
            perm[i] = j
            used_r.add(i)
            used_c.add(j)
            if len(used_r) == M:
                break
    cost_ms = (time.perf_counter() - t0) * 1000.0
    perm_t = torch.tensor(perm, device=C_new.device, dtype=torch.long)
    return C_new[perm_t], cost_ms


@torch.no_grad()
def compute_token_overlap(X_sub: torch.Tensor, C_ref: torch.Tensor, C_cand: torch.Tensor) -> float:
    c_ref_half = 0.5 * (C_ref * C_ref).sum(dim=1)
    c_cand_half = 0.5 * (C_cand * C_cand).sum(dim=1)
    a_ref = torch.argmax(torch.addmm(-c_ref_half, X_sub, C_ref.T), dim=1)
    a_cand = torch.argmax(torch.addmm(-c_cand_half, X_sub, C_cand.T), dim=1)
    return float((a_ref == a_cand).float().mean().item()) * 100.0


@torch.no_grad()
def evaluate_quantization(X: torch.Tensor, C: torch.Tensor) -> float:
    chunk_size = 65536
    total_sq = 0.0
    c_half = 0.5 * (C * C).sum(dim=1)
    n = X.shape[0]
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        scores = torch.addmm(-c_half, chunk, C.T)
        best = torch.max(scores, dim=1).values
        x_sq = (chunk * chunk).sum(dim=1)
        total_sq += torch.clamp(x_sq - 2.0 * best, min=0.0).sum().item()
    return total_sq / n


# =====================================================================
# 5. The 5 Diverse Stress Task Generators (Base & Drift)
# =====================================================================

def generate_task_base_and_drift(task_id: int, N: int, D: int, K: int, drift_scale: float = 0.05):
    torch.manual_seed(42)

    if task_id == 1:
        # Task 1: Isolated Blobs + 20 Outliers
        centers = torch.randn(K, D, device=device) * 4.0
        pts_per_k = N // K
        X_base = torch.empty((N + 20, D), device=device, dtype=torch.float32)
        X_drift = torch.empty((N + 20, D), device=device, dtype=torch.float32)
        center_drift = torch.randn_like(centers) * (drift_scale * 4.0)
        centers_drifted = centers + center_drift

        for i in range(K):
            noise = torch.randn(pts_per_k, D, device=device)
            X_base[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers[i]
            X_drift[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers_drifted[i] + torch.randn_like(noise) * 0.05

        torch.randn(20, D, device=device, out=X_base[N:]).mul_(35.0)
        torch.randn(20, D, device=device, out=X_drift[N:]).mul_(35.0)
        return "Task 1: Isolated Blobs + 20 Outliers", X_base, X_drift

    elif task_id == 2:
        # Task 2: L2-Normalized Manifold
        latent_rank, num_topics = 64, 256
        Q, _ = torch.linalg.qr(torch.randn(D, latent_rank, device=device))
        topic_centers = F.normalize(torch.randn(num_topics, latent_rank, device=device), dim=1)
        probs = (torch.arange(num_topics, device=device, dtype=torch.float32) + 2.0) ** -1.1
        probs /= probs.sum()

        R = torch.eye(latent_rank, device=device)
        c, s = math.cos(drift_scale * 1.5), math.sin(drift_scale * 1.5)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c

        ids = torch.multinomial(probs, N, replacement=True)
        noise = torch.randn(N, latent_rank, device=device) * 0.42

        Z_base = topic_centers[ids] + noise
        X_base = F.normalize(Z_base @ Q.T + torch.randn(N, D, device=device) * 0.015, dim=-1)

        Z_drift = (topic_centers @ R)[ids] + noise + torch.randn_like(noise) * 0.02
        X_drift = F.normalize(Z_drift @ Q.T + torch.randn(N, D, device=device) * 0.015, dim=-1)
        return "Task 2: L2-Normalized Manifold (Zipfian Hubs)", X_base, X_drift

    elif task_id == 3:
        # Task 3: Rotated Anisotropic Subspace
        Q, _ = torch.linalg.qr(torch.randn(D, D, device=device))
        eig_scales = 10.0 * (0.985 ** torch.arange(D, device=device, dtype=torch.float32)) + 0.15
        centers = (torch.randn(K, D, device=device) * eig_scales) @ Q
        centers_drifted = centers + (torch.randn_like(centers) * eig_scales * drift_scale) @ Q

        pts_per_k = N // K
        X_base = torch.empty((N, D), device=device, dtype=torch.float32)
        X_drift = torch.empty((N, D), device=device, dtype=torch.float32)
        local_scales = eig_scales * 0.25

        for i in range(K):
            noise = (torch.randn(pts_per_k, D, device=device) * local_scales) @ Q
            X_base[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers[i]
            X_drift[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers_drifted[i]

        return "Task 3: Rotated Anisotropic Subspace", X_base, X_drift

    elif task_id == 4:
        # Task 4: 96% Core + 4% Heavy Tail
        n_core, n_noise = int(0.96 * N), N - int(0.96 * N)
        centers = torch.randn(K, D, device=device) * 3.0
        centers_drifted = centers + torch.randn_like(centers) * (drift_scale * 3.0)

        pts_per_k = n_core // K
        X_base = torch.empty((N, D), device=device, dtype=torch.float32)
        X_drift = torch.empty((N, D), device=device, dtype=torch.float32)

        for i in range(K):
            noise = torch.randn(pts_per_k, D, device=device)
            X_base[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers[i]
            X_drift[i * pts_per_k : (i + 1) * pts_per_k] = noise + centers_drifted[i]

        torch.randn(n_noise, D, device=device, out=X_base[n_core:]).mul_(10.0)
        torch.randn(n_noise, D, device=device, out=X_drift[n_core:]).mul_(10.0)
        return "Task 4: 96% Core + 4% Heavy-Tail Clutter", X_base, X_drift

    else:
        # Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)
        n_giants = int(0.75 * N)
        n_dwarfs = N - n_giants
        num_dwarfs = max(1, K - 2)
        g_pts = n_giants // 2
        d_pts = n_dwarfs // num_dwarfs

        centers_g = torch.randn(2, D, device=device) * 4.0
        centers_d = torch.randn(num_dwarfs, D, device=device) * 4.0
        cg_drift = centers_g + torch.randn_like(centers_g) * (drift_scale * 4.0)
        cd_drift = centers_d + torch.randn_like(centers_d) * (drift_scale * 4.0)

        X_base = torch.empty((N, D), device=device, dtype=torch.float32)
        X_drift = torch.empty((N, D), device=device, dtype=torch.float32)

        for g in range(2):
            s, e = g * g_pts, (g + 1) * g_pts
            noise = torch.randn(e - s, D, device=device) * 3.2
            X_base[s:e] = noise + centers_g[g]
            X_drift[s:e] = noise + cg_drift[g]

        for d in range(num_dwarfs):
            s = n_giants + d * d_pts
            e = n_giants + (d + 1) * d_pts if d < num_dwarfs - 1 else N
            noise = torch.randn(e - s, D, device=device) * 0.6
            X_base[s:e] = noise + centers_d[d]
            X_drift[s:e] = noise + cd_drift[d]

        return "Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)", X_base, X_drift


# =====================================================================
# 6. Benchmark Orchestrator
# =====================================================================

def run_benchmark():
    N, D, K, SUB_N = 320000, 1024, 64, 16384

    # Warmup at exact benchmark (SUB_N, D, K) so CUDA graphs are pre-captured before timing
    dummy = torch.randn(SUB_N, D, device=device)
    c_dum = torch.randn(K, D, device=device)
    _ = run_engine_v11_triton(dummy, K, sub_n=SUB_N)
    _ = triton_reproject_lloyd_step(dummy, c_dum, sub_n=SUB_N)
    _ = triton_reproject_engine_step(dummy, c_dum, sub_n=SUB_N, steps=1, polish=True)
    _ = run_lloyd_cublas(dummy, K, iters=1)
    _ = unscramble_hungarian_cosine(c_dum, c_dum)
    _ = unscramble_greedy_bipartite(c_dum, c_dum)
    torch.cuda.synchronize()

    print("=" * 118)
    print(f" GENUINE ENGINE V11 MULTI-TASK DRIFT & REPROJECTION BENCHMARK (N={N:,} | D={D} | K={K})")
    print(f" Hardware: {torch.cuda.get_device_name(0)} | Base Fitter: Engine v11 (nosync)")
    print("=" * 118)

    methods = [
        ("Triton Lloyd Reproj (1 pass, 16k)",   "Warm", lambda x, c: triton_reproject_lloyd_step(x, c, sub_n=SUB_N)),
        ("Triton Lloyd Reproj (2 passes, 16k)", "Warm", lambda x, c: triton_reproject_lloyd_step(x, triton_reproject_lloyd_step(x, c, sub_n=SUB_N), sub_n=SUB_N)),
        ("Triton Engine Reproj (1 step, 16k)",  "Warm", lambda x, c: triton_reproject_engine_step(x, c, sub_n=SUB_N, steps=1, polish=True)),
        ("Triton Engine Reproj (2 pass, 16k)",  "Warm", lambda x, c: triton_reproject_engine_step(x, triton_reproject_engine_step(x, c, sub_n=SUB_N, steps=1, polish=True), sub_n=SUB_N, steps=1, polish=True)),
        ("CuBLAS Lloyd Warm (1 pass, full)",    "Warm", lambda x, c: run_lloyd_cublas(x, K, iters=1, init_C=c)),
        ("Engine v11 Cold (from scratch)",      "Cold", lambda x, c: run_engine_v11_triton(x, K, sub_n=SUB_N)),
        ("e-means Cold (1.0 ep)",               "Cold", lambda x, c: run_original_emeans(x, K, epochs=1.0, init_C=None)),
        ("CuBLAS Lloyd Cold (15 iters)",        "Cold", lambda x, c: run_lloyd_cublas(x, K, iters=15, init_C=None)),
    ]

    for task_id in range(1, 6):
        title, X_base, X_drift = generate_task_base_and_drift(task_id, N, D, K, drift_scale=0.06)

        # 1. Base Fit with ENGINE V11!
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C_base = run_engine_v11_triton(X_base, K, sub_n=SUB_N)
        torch.cuda.synchronize()
        base_time = (time.perf_counter() - t0) * 1000.0

        base_mse = evaluate_quantization(X_base, C_base)
        frozen_mse = evaluate_quantization(X_drift, C_base)

        print(f"\n>>> {title}")
        print(f"    v11 Initial Fit (t=0): {base_time:.1f} ms | Base MSE: {base_mse:.4f}")
        print(f"    Frozen v11 Codebook on X_drift: MSE = {frozen_mse:.4f} (No update)")
        print("-" * 118)
        print(
            f"{'Method':<36} | {'Type':<4} | {'Fit ms':<7} | {'MSE':<9} | "
            f"{'Raw Cos':<8} | {'Raw Ovr%':<8} | {'Hung Cos (ms)':<16} | {'Greedy (ms)'}"
        )
        print("-" * 118)

        audit_sub = X_drift[:SUB_N]

        for name, m_type, fn in methods:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            C_cand = fn(X_drift, C_base)
            torch.cuda.synchronize()
            fit_ms = (time.perf_counter() - t0) * 1000.0

            adapted_mse = evaluate_quantization(X_drift, C_cand)

            raw_cos = float(F.cosine_similarity(C_base, C_cand, dim=-1).mean().item())
            raw_ovr = compute_token_overlap(audit_sub, C_base, C_cand)

            C_hung_cos, t_hung_cos = unscramble_hungarian_cosine(C_base, C_cand)
            hung_cos = float(F.cosine_similarity(C_base, C_hung_cos, dim=-1).mean().item())

            C_greedy, t_greedy = unscramble_greedy_bipartite(C_base, C_cand)
            greedy_cos = float(F.cosine_similarity(C_base, C_greedy, dim=-1).mean().item())

            raw_cos_str = f"{raw_cos:+.3f}" if raw_cos > 0.3 else "SCRAMBLED"
            raw_ovr_str = f"{raw_ovr:4.1f}%" if raw_cos > 0.3 else "0.0%"
            hung_str = f"{hung_cos:+.3f} ({t_hung_cos:.2f}ms)"
            greedy_str = f"{greedy_cos:+.3f} ({t_greedy:.2f}ms)"

            print(
                f"{name:<36} | {m_type:<4} | {fit_ms:<7.2f} | {adapted_mse:<9.4f} | "
                f"{raw_cos_str:<8} | {raw_ovr_str:<8} | {hung_str:<16} | {greedy_str}"
            )

        del X_base, X_drift
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_benchmark()