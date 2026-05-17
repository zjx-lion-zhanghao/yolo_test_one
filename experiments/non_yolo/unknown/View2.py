# View2_fixed.py
import os, json, math, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from ortools.sat.python import cp_model
from tqdm import tqdm

# ===================== Problem constants =====================
TMIN, TMAX = 69, 187
TAU_T = TMAX - TMIN

D = 100_000_000
QE, QR = 2983, 125
MMAX, NMAX = 60, 730

cE = 1.8
ck = np.array([1.99, 1.47, 1.34, 1.37, 1.53, 1.69, 1.10, 1.19, 1.55, 1.55], dtype=float)

C0_MIN = 6.06 * 10000
C0_MAX = 120.5 * 10000
TAU_C_FIXED = 120.5 * 10000

SCALE = 100  # money scaling
cE_i = int(round(cE * SCALE))
ck_i = np.array([int(round(x * SCALE)) for x in ck], dtype=np.int64)
ck_min_i, ck_max_i = int(ck_i.min()), int(ck_i.max())


# ===================== Utilities =====================
def uT_np(T: int) -> float:
    return float(np.log(1.0 + abs(T - TMIN) / TAU_T))

def uT_torch(T):
    return torch.log1p(torch.abs(T - TMIN) / TAU_T)

def uC_torch(absdiff, tauC):
    return torch.log1p(absdiff / tauC)

def recover_Nk_exact(T: int, N_total: int, WR_scaled: int) -> Optional[np.ndarray]:
    """
    Meet-in-the-middle exact reconstruction for Nk such that:
      sum_k Nk = N_total
      sum_k ck_i[k]*Nk = WR_scaled
      0 <= Nk <= T*NMAX
    """
    U = T * NMAX
    costs = ck_i.tolist()
    left = list(range(5))
    right = list(range(5, 10))

    left_map = {}

    def dfs_left(i, N, WR, Nk):
        if i == 5:
            left_map.setdefault((N, WR), Nk.copy())
            return
        k = left[i]; c = costs[k]
        max_n = min(U, N_total - N)
        for n in range(max_n + 1):
            WR2 = WR + n * c
            if WR2 > WR_scaled:
                break
            Nk.append(n)
            dfs_left(i + 1, N + n, WR2, Nk)
            Nk.pop()

    dfs_left(0, 0, 0, [])

    def dfs_right(i, N, WR, Nk):
        if i == 5:
            key = (N_total - N, WR_scaled - WR)
            if key in left_map:
                return np.array(left_map[key] + Nk, dtype=int)
            return None
        k = right[i]; c = costs[k]
        max_n = min(U, N_total - N)
        for n in range(max_n + 1):
            WR2 = WR + n * c
            if WR2 > WR_scaled:
                break
            Nk.append(n)
            out = dfs_right(i + 1, N + n, WR2, Nk)
            if out is not None:
                return out
            Nk.pop()
        return None

    return dfs_right(0, 0, 0, [])

def build_yearly_plan(T: int, M: int, Nk: np.ndarray):
    m = np.zeros((3, T), dtype=int)
    n = np.zeros((10, T), dtype=int)

    slots = 3 * T
    base = M // slots
    rem = M % slots
    if base > MMAX:
        raise ValueError("M exceeds per-slot capacity")
    m[:] = base
    idx = 0
    while rem > 0:
        j = idx // 3; i = idx % 3
        if m[i, j] < MMAX:
            m[i, j] += 1
            rem -= 1
        idx = (idx + 1) % slots

    for k in range(10):
        total = int(Nk[k])
        base = total // T
        rem = total % T
        if base > NMAX:
            raise ValueError("Nk exceeds per-year cap")
        n[k, :] = base
        for j in range(T):
            if rem <= 0:
                break
            if n[k, j] < NMAX:
                n[k, j] += 1
                rem -= 1
    return m, n


# ===================== Output struct =====================
@dataclass
class BestSol:
    alpha: float
    beta: float
    eps: float
    C0: float
    tauC: float
    T: int
    N: int
    M: int
    WR: float
    W: float
    Q: int
    B: float


# ===================== Strict solver (CP-SAT) =====================
def solve_one_strict(alpha: float, C0: float, eps: float, tauC: float,
                     time_limit_s_per_T: float = 0.25,
                     workers: int = 1) -> Optional[BestSol]:
    """
    Return best solution for this (alpha,C0,eps); if infeasible for all T, return None.
    Constraints: D <= Q <= (1+eps)D.
    """

    C0_i = int(round(C0 * SCALE))
    Q_lo = int(D)
    Q_hi = int(math.floor((1.0 + eps) * D))

    best: Optional[BestSol] = None

    for T in range(TMIN, TMAX + 1):
        M_cap = 3 * T * MMAX
        N_cap = 10 * T * NMAX

        # fast load-window feasibility against capacities
        Q_max_possible = M_cap * QE + N_cap * QR
        if Q_max_possible < Q_lo:
            continue

        model = cp_model.CpModel()

        M = model.NewIntVar(0, M_cap, "M")
        N = model.NewIntVar(0, N_cap, "N")
        Nk = [model.NewIntVar(0, T * NMAX, f"N{k}") for k in range(10)]
        model.Add(sum(Nk) == N)

        Qexpr = QE * M + QR * N
        model.Add(Qexpr >= Q_lo)
        model.Add(Qexpr <= Q_hi)

        W_ub = cE_i * M_cap + ck_max_i * (10 * T * NMAX)
        W = model.NewIntVar(0, int(W_ub), "W")
        model.Add(W == cE_i * M + sum(int(ck_i[k]) * Nk[k] for k in range(10)))

        diff = model.NewIntVar(-int(W_ub) - abs(C0_i), int(W_ub) + abs(C0_i), "diff")
        model.Add(diff == W - C0_i)
        Delta = model.NewIntVar(0, int(W_ub) + abs(C0_i), "Delta")
        model.AddAbsEquality(Delta, diff)

        model.Minimize(Delta)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_s_per_T)
        solver.parameters.num_search_workers = int(workers)
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue

        Mv = int(solver.Value(M))
        Nv = int(solver.Value(N))
        Wv_i = int(solver.Value(W))

        Nk_val = [int(solver.Value(Nk[k])) for k in range(10)]
        WR_i = int(sum(int(ck_i[k]) * Nk_val[k] for k in range(10)))

        Wv = Wv_i / SCALE
        WRv = WR_i / SCALE
        Qv = Mv * QE + Nv * QR
        absdiff = abs(Wv - C0)

        Bv = float(alpha * uT_np(T) + (1.0 - alpha) * math.log(1.0 + absdiff / float(tauC)))

        sol = BestSol(
            alpha=float(alpha),
            beta=float(1.0 - alpha),
            eps=float(eps),
            C0=float(C0),
            tauC=float(tauC),
            T=int(T),
            N=int(Nv),
            M=int(Mv),
            WR=float(WRv),
            W=float(Wv),
            Q=int(Qv),
            B=float(Bv),
        )
        if (best is None) or (sol.B < best.B - 1e-15):
            best = sol

    return best


# ===================== Safe LB for pruning (Torch) =====================
def compute_LB_grid_torch(alphas, eps_list, C0_list, tauC=TAU_C_FIXED, device="cuda"):
    """
    Safe LB for load window D <= Q <= (1+eps)D.
    See earlier explanation: relaxation envelope + dist-to-interval.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    eps_list = np.asarray(eps_list, dtype=np.float64)
    C0_list = np.asarray(C0_list, dtype=np.float64)

    T_vals = torch.arange(TMIN, TMAX + 1, device=device, dtype=torch.float32)
    T_int = T_vals.to(torch.int64)
    uT_vals = uT_torch(T_vals)

    M_cap = (3 * T_int * MMAX).to(torch.float32)
    N_cap = (10 * T_int * NMAX).to(torch.float32)
    M_cap_T = M_cap[:, None, None]
    N_cap_T = N_cap[:, None, None]

    eps_t = torch.tensor(eps_list, device=device, dtype=torch.float32)[None, :, None]  # 1×E×1
    Q_hi = (1.0 + eps_t) * float(D)  # 1×E×1
    Q_lo = float(D)

    Qcap = (M_cap_T * QE + N_cap_T * QR)  # T×1×1
    feasible_mask = (Qcap >= Q_lo).expand(T_vals.shape[0], eps_t.shape[1], 1)  # T×E×1

    elev_cp = float(cE / QE)

    def cost_for_targetQ(targetQ, rocket_cost_per_unit):
        rock_cp = rocket_cost_per_unit / QR
        if rock_cp <= elev_cp:
            N_use = torch.minimum(N_cap_T, targetQ / QR)
            Q_left = torch.clamp(targetQ - N_use * QR, min=0.0)
            M_use = torch.minimum(M_cap_T, Q_left / QE)
        else:
            M_use = torch.minimum(M_cap_T, targetQ / QE)
            Q_left = torch.clamp(targetQ - M_use * QE, min=0.0)
            N_use = torch.minimum(N_cap_T, Q_left / QR)
        return cE * M_use + rocket_cost_per_unit * N_use

    Qlo_TE = torch.full((T_vals.shape[0], eps_t.shape[1], 1), float(D), device=device, dtype=torch.float32)
    Qhi_TE = Q_hi.expand(T_vals.shape[0], -1, -1)

    INF = 1e30
    Wlo = torch.full((T_vals.shape[0], eps_t.shape[1], 1), INF, device=device, dtype=torch.float32)
    Whi = torch.full((T_vals.shape[0], eps_t.shape[1], 1), -INF, device=device, dtype=torch.float32)

    for Qtarget in (Qlo_TE, Qhi_TE):
        Wa = cost_for_targetQ(Qtarget, ck_min_i / SCALE)
        Wb = cost_for_targetQ(Qtarget, ck_max_i / SCALE)
        Wlo = torch.minimum(Wlo, torch.minimum(Wa, Wb))
        Whi = torch.maximum(Whi, torch.maximum(Wa, Wb))

    Wlo = torch.where(feasible_mask, Wlo, torch.full_like(Wlo, INF))
    Whi = torch.where(feasible_mask, Whi, torch.full_like(Whi, -INF))

    C0_t = torch.tensor(C0_list, device=device, dtype=torch.float32)[None, None, :]  # 1×1×C
    C0_b = C0_t.expand(Wlo.shape[0], Wlo.shape[1], C0_t.shape[2])
    Wlo_b = Wlo.expand(-1, -1, C0_t.shape[2])
    Whi_b = Whi.expand(-1, -1, C0_t.shape[2])

    infeas = (Wlo_b > Whi_b)
    dist = torch.where(
        infeas,
        torch.full_like(C0_b, INF),
        torch.where(
            (C0_b >= Wlo_b) & (C0_b <= Whi_b),
            torch.zeros_like(C0_b),
            torch.minimum(torch.abs(C0_b - Wlo_b), torch.abs(C0_b - Whi_b))
        )
    )

    uT_T = uT_vals[:, None, None]
    uC = uC_torch(dist, float(tauC))  # T×E×C

    alphas_t = torch.tensor(alphas, device=device, dtype=torch.float32)[:, None, None]  # A×1×1
    A = alphas_t.shape[0]
    Tn, En, Cn = uC.shape

    uT_b = uT_T.view(1, Tn, 1, 1)
    uC_b = uC.view(1, Tn, En, Cn)
    alpha_b = alphas_t.view(A, 1, 1, 1)

    LB_T = alpha_b * uT_b + (1.0 - alpha_b) * uC_b
    LB = torch.amin(LB_T, dim=1)  # A×E×C
    return LB


# ===================== Multiprocess worker =====================
def _strict_worker(args):
    ai, ei, ci, alpha, eps, C0, tauC, time_limit_s_per_T = args
    sol = solve_one_strict(alpha, C0, eps, tauC, time_limit_s_per_T=time_limit_s_per_T, workers=1)
    return ai, ei, ci, sol  # sol can be None


# ===================== Main pipeline =====================
def run(out_dir="out_run",
        C0_points=50,
        eps_points=5,
        alpha_list=None,
        max_workers=6,
        time_limit_s_per_T=0.25):

    os.makedirs(out_dir, exist_ok=True)

    if alpha_list is None:
        alpha_list = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)

    C0_list = np.linspace(C0_MIN, C0_MAX, C0_points).astype(np.float64)
    eps_list = np.linspace(1e-5, 1e-3, eps_points).astype(np.float64)

    # ---- Stage 1: LB on full grid ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1] computing LB on full grid using device={device} ...")
    t0 = time.time()
    LB = compute_LB_grid_torch(alpha_list, eps_list, C0_list, device=device).cpu().numpy()
    print(f"[Stage 1] LB done in {time.time()-t0:.2f}s, shape={LB.shape}")

    A, E, C = LB.shape
    B_strict = np.full((A, E, C), np.inf, dtype=np.float64)

    flat = []
    for ai, a in enumerate(alpha_list):
        for ei, eps in enumerate(eps_list):
            for ci, C0 in enumerate(C0_list):
                flat.append((float(LB[ai, ei, ci]), ai, ei, ci, float(a), float(eps), float(C0)))
    flat.sort(key=lambda x: x[0])

    INF_LB = 1e25
    best_sol: Optional[BestSol] = None
    best_B = float("inf")

    print(f"[Stage 2] strict solving with pruning, max_workers={max_workers} ...")

    submitted = 0
    completed = 0
    pruned = 0
    skipped_infeas_lb = 0
    strict_infeasible = 0
    idx = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}

        def try_submit_one():
            nonlocal idx, submitted, pruned, skipped_infeas_lb
            while idx < len(flat):
                lb, ai, ei, ci, a, eps, C0 = flat[idx]
                idx += 1

                if lb >= INF_LB:
                    skipped_infeas_lb += 1
                    continue
                if lb >= best_B:
                    pruned += 1
                    continue

                fut = ex.submit(_strict_worker, (ai, ei, ci, a, eps, C0, float(TAU_C_FIXED), float(time_limit_s_per_T)))
                futures[fut] = (lb, ai, ei, ci)
                submitted += 1
                return True
            return False

        # seed
        for _ in range(max_workers * 4):
            if not try_submit_one():
                break

        pbar = tqdm(total=len(flat), desc="Grid points (done/pruned/skip)", ncols=120)
        pbar.update(pruned + skipped_infeas_lb)

        while True:
            next_lb = flat[idx][0] if idx < len(flat) else float("inf")

            # proven optimal when: have best, nothing running, and remaining LB can't beat best_B
            if best_sol is not None and next_lb >= best_B and len(futures) == 0:
                break

            while len(futures) < max_workers * 8:
                if not try_submit_one():
                    break

            if len(futures) == 0:
                if idx >= len(flat):
                    break
                continue

            # wait one completion
            done_one = None
            for fut in as_completed(list(futures.keys()), timeout=None):
                done_one = fut
                break

            lb, ai, ei, ci = futures.pop(done_one)
            ai2, ei2, ci2, sol = done_one.result()
            assert (ai, ei, ci) == (ai2, ei2, ci2)

            completed += 1
            pbar.update(1)

            if sol is None:
                strict_infeasible += 1
                # keep B_strict as +inf
            else:
                B_strict[ai, ei, ci] = sol.B
                if sol.B < best_B:
                    best_B = sol.B
                    best_sol = sol

            pbar.set_postfix({
                "best_B": f"{best_B:.4g}" if best_sol else "inf",
                "next_LB": f"{next_lb:.4g}",
                "run": len(futures),
                "done": completed,
                "pruned": pruned,
                "skipInfLB": skipped_infeas_lb,
                "strictInfeas": strict_infeasible
            })

        pbar.close()

    if best_sol is None:
        raise RuntimeError("No feasible strict solution found anywhere on the grid (load window may be too tight).")

    # ---- Recover Nk and yearly plan for best ----
    WR_scaled = int(round(best_sol.WR * SCALE))
    Nk = recover_Nk_exact(best_sol.T, best_sol.N, WR_scaled)
    if Nk is None:
        raise RuntimeError("Failed to recover Nk from (T,N,WR). Consider storing Nk directly.")
    m, n = build_yearly_plan(best_sol.T, best_sol.M, Nk)

    # ---- Save outputs ----
    LBmin = LB.min(axis=0)  # E×C
    np.save(os.path.join(out_dir, "LBmin_alpha.npy"), LBmin)
    np.save(os.path.join(out_dir, "B_strict.npy"), B_strict)

    best_payload = {
        "solution": asdict(best_sol),
        "Nk": Nk.tolist(),
        "m_3xT": m.tolist(),
        "n_10xT": n.tolist(),
        "meta": {
            "strict_solved_points": int(np.isfinite(B_strict).sum()),
            "strict_infeasible_points": int(strict_infeasible),
            "grid_total_points": int(A * E * C),
            "max_workers": int(max_workers),
            "time_limit_s_per_T": float(time_limit_s_per_T),
            "load_constraint": "D <= Q <= (1+eps)D",
            "heatmap_is": "min_alpha LB (safe lower bound) over full grid; red X is strict global best point."
        }
    }
    with open(os.path.join(out_dir, "best_solution.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    # ---- Plot Heatmap A1 (LBmin) ----
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=140)
    im = ax.imshow(LBmin, origin="lower", aspect="auto",
                   extent=[C0_list.min(), C0_list.max(), eps_list.min(), eps_list.max()])
    ax.scatter([best_sol.C0], [best_sol.eps], c="red", s=50, marker="x", linewidths=2,
               label=f"Strict best: alpha={best_sol.alpha:.1f}, B={best_sol.B:.4g}")
    ax.set_xlabel("C0 (亿美元)")
    ax.set_ylabel("eps")
    ax.set_title("Heatmap A1: min_alpha LB (safe lower bound); strict global best marked")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("min_alpha LB")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_A1.png"), bbox_inches="tight")
    plt.show()

    print(f"[DONE] best_B={best_sol.B:.6g}, best at (alpha={best_sol.alpha}, eps={best_sol.eps}, C0={best_sol.C0})")
    print(f"Saved: {out_dir}/heatmap_A1.png, {out_dir}/best_solution.json")
    return best_payload


if __name__ == "__main__":
    run(
        out_dir="out_run",
        C0_points=50,
        eps_points=5,
        alpha_list=np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1),
        max_workers=6,
        time_limit_s_per_T=0.25
    )
