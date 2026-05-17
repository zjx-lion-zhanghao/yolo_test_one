# sensitivity_gpu_and_plot.py
import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch

# 可选：用于光滑曲面（CPU侧插值；画图本身也在CPU）
from scipy.interpolate import RectBivariateSpline, RBFInterpolator


# -----------------------------
# Part A: Run GPU sensitivity (reuse your basic.py)
# -----------------------------
def run_gpu_sensitivity_via_basic(
    out_csv="sensitivity_summary.csv",
    mode="full",
):
    """
    复用 basic.py 的 GPU 灵敏性分析。
    你的 basic.py 末尾支持:
      python basic.py test
      python basic.py full
    这里直接 import basic 并调用其 main_gpu/quick_test_gpu。
    运行完成后，尝试在当前目录找到 basic.py 输出的 csv；
    若 basic.py 已经输出 out_csv，则直接返回 out_csv。
    """

    basic = importlib.import_module("basic")

    # GPU可用性检查
    print("---- GPU CHECK (torch) ----")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("torch cuda:", torch.version.cuda)
    print("---------------------------")

    # 运行 basic.py 里的流程
    if mode == "test":
        if hasattr(basic, "quick_test_gpu"):
            basic.quick_test_gpu()
        else:
            raise AttributeError("basic.py 中找不到 quick_test_gpu()")
    elif mode == "full":
        if hasattr(basic, "main_gpu"):
            basic.main_gpu()
        else:
            raise AttributeError("basic.py 中找不到 main_gpu()")
    else:
        raise ValueError("mode must be 'test' or 'full'")

    # basic.py 可能输出的文件名不一定叫 sensitivity_summary.csv
    # 我们做一个兜底：找最近修改的csv，并优先匹配包含 alpha/C0/B 等字段的
    if os.path.exists(out_csv):
        print(f"[OK] Found output CSV: {out_csv}")
        return out_csv

    # 兜底：从当前目录找csv
    cands = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if not cands:
        raise FileNotFoundError("未找到任何CSV输出文件。请确认 basic.py 是否写出了结果CSV。")

    # 按修改时间排序，取最新
    cands.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    for f in cands:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = set(df.columns)
        # 你画图脚本期望 alpha, C0_yi, B_star（或 basic 的 alpha,C0,B）
        if ("alpha" in cols) and (("C0_yi" in cols) or ("C0" in cols)) and (("B_star" in cols) or ("B" in cols)):
            print(f"[OK] Using latest matching CSV: {f}")
            # 规范化列名成 alpha, C0_yi, B_star 方便后续绘图
            df2 = normalize_columns(df)
            df2.to_csv(out_csv, index=False)
            print(f"[OK] Normalized and saved as: {out_csv}")
            return out_csv

    raise FileNotFoundError(
        f"basic.py 运行结束，但没有找到包含 alpha/C0/B 的CSV。当前目录CSV候选：{cands[:10]}"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 basic.py 可能输出的列名统一为:
      alpha, C0_yi, B_star
    并尽量保留其它列（T,W等）。
    """
    out = df.copy()
    # C0
    if "C0_yi" not in out.columns:
        if "C0" in out.columns:
            out = out.rename(columns={"C0": "C0_yi"})
    # B
    if "B_star" not in out.columns:
        if "B" in out.columns:
            out = out.rename(columns={"B": "B_star"})
    # alpha already
    return out


# -----------------------------
# Part B: Smooth surface + find min + plot
# -----------------------------
def load_grid(csv_path="sensitivity_summary.csv"):
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # 去重/排序
    alphas = np.sort(df["alpha"].unique())
    C0s = np.sort(df["C0_yi"].unique())

    pivot = df.pivot(index="alpha", columns="C0_yi", values="B_star")
    B = pivot.values
    assert B.shape == (len(alphas), len(C0s))
    return alphas, C0s, B, df


def smooth_surface_spline(alphas, C0s, B, fine_a=201, fine_c=301):
    spline = RectBivariateSpline(alphas, C0s, B, kx=3, ky=3, s=0.0)
    a_f = np.linspace(alphas.min(), alphas.max(), fine_a)
    c_f = np.linspace(C0s.min(), C0s.max(), fine_c)
    B_f = spline(a_f, c_f)

    idx = np.unravel_index(np.argmin(B_f), B_f.shape)
    a_min, c_min, b_min = a_f[idx[0]], c_f[idx[1]], float(B_f[idx])
    return a_f, c_f, B_f, (a_min, c_min, b_min)


def smooth_surface_rbf(alphas, C0s, B, fine_a=201, fine_c=301, kernel="thin_plate_spline", smooth=1e-6):
    A, C = np.meshgrid(alphas, C0s, indexing="ij")
    X = np.column_stack([A.ravel(), C.ravel()])
    y = B.ravel()

    rbf = RBFInterpolator(X, y, kernel=kernel, smoothing=smooth)

    a_f = np.linspace(alphas.min(), alphas.max(), fine_a)
    c_f = np.linspace(C0s.min(), C0s.max(), fine_c)
    Af, Cf = np.meshgrid(a_f, c_f, indexing="ij")
    Xf = np.column_stack([Af.ravel(), Cf.ravel()])
    B_f = rbf(Xf).reshape(fine_a, fine_c)

    idx = np.unravel_index(np.argmin(B_f), B_f.shape)
    a_min, c_min, b_min = a_f[idx[0]], c_f[idx[1]], float(B_f[idx])
    return a_f, c_f, B_f, (a_min, c_min, b_min)


def plot_3d_surface(a_f, c_f, B_f, minpt, title):
    Af, Cf = np.meshgrid(a_f, c_f, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Af, Cf, B_f, cmap=cm.jet, linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, shrink=0.6, aspect=18, label="B")

    a_min, c_min, b_min = minpt
    ax.scatter([a_min], [c_min], [b_min], color="k", s=55, depthshade=False, label="argmin (smoothed)")

    ax.set_xlabel("alpha")
    ax.set_ylabel("C0 (亿美元)")
    ax.set_zlabel("B")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 用法：
    #   python sensitivity_gpu_and_plot.py full   # 跑完整GPU灵敏性 + 画图
    #   python sensitivity_gpu_and_plot.py test   # 跑快速GPU测试 + 画图
    #   python sensitivity_gpu_and_plot.py plot   # 只读取已有 sensitivity_summary.csv 画图

    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode in ("full", "test"):
        csv_path = run_gpu_sensitivity_via_basic(out_csv="sensitivity_summary.csv", mode=mode)
    elif mode == "plot":
        csv_path = "sensitivity_summary.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError("未找到 sensitivity_summary.csv；请先运行 full/test 或把csv放到当前目录。")
    else:
        raise ValueError("mode must be: full | test | plot")

    # 画图 + 最小点
    alphas, C0s, B, df = load_grid(csv_path)

    # A) 样条（规则网格推荐）
    a_f, c_f, B_f, minpt = smooth_surface_spline(alphas, C0s, B, fine_a=201, fine_c=301)
    print("[Spline] min at alpha=%.4f, C0=%.2f(亿美元), B=%.8f" % minpt)
    plot_3d_surface(a_f, c_f, B_f, minpt, "B(alpha, C0) - RectBivariateSpline")

    # B) RBF（更平滑）
    a_f2, c_f2, B_f2, minpt2 = smooth_surface_rbf(alphas, C0s, B, fine_a=201, fine_c=301,
                                                  kernel="thin_plate_spline", smooth=1e-6)
    print("[RBF]   min at alpha=%.4f, C0=%.2f(亿美元), B=%.8f" % minpt2)
    plot_3d_surface(a_f2, c_f2, B_f2, minpt2, "B(alpha, C0) - RBFInterpolator")


if __name__ == "__main__":
    main()
