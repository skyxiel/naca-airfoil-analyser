"""
NACA 4-digit airfoil generator with vortex panel method for
pressure distribution and lift coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ── NACA 4-digit geometry ────────────────────────────────────────────────────

def naca4_thickness(t: float, x: np.ndarray) -> np.ndarray:
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def naca4_camber(m: float, p: float, x: np.ndarray):
    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    dyc = np.where(
        x < p,
        2 * m / p**2 * (p - x),
        2 * m / (1 - p)**2 * (p - x),
    )
    return yc, dyc


def generate_naca4(designation: str, n_points: int = 100):
    m = int(designation[0]) / 100
    p = int(designation[1]) / 10
    t = int(designation[2:]) / 100

    # cosine spacing for better leading-edge resolution
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    yt = naca4_thickness(t, x)

    if m == 0 or p == 0:
        xu, xl = x, x
        yu, yl = yt, -yt
    else:
        yc, dyc = naca4_camber(m, p, x)
        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    # panel-method ordering: upper surface TE→LE then lower surface LE→TE
    xp = np.concatenate([xu[::-1], xl[1:]])
    yp = np.concatenate([yu[::-1], yl[1:]])
    return xp, yp

# ── Vortex panel method ─────────────────────────────────────────────────────

@dataclass
class PanelResult:
    cp: np.ndarray
    cl: float
    xm: np.ndarray
    ym: np.ndarray
    gamma: np.ndarray


def vortex_panel_method(xp: np.ndarray, yp: np.ndarray, alpha_deg: float) -> PanelResult:
    alpha = np.radians(alpha_deg)
    n = len(xp) - 1  # number of panels

    # panel geometry
    xm = 0.5 * (xp[:-1] + xp[1:])
    ym = 0.5 * (yp[:-1] + yp[1:])
    dx = xp[1:] - xp[:-1]
    dy = yp[1:] - yp[:-1]
    s = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # influence coefficients
    A = np.zeros((n + 1, n + 1))

    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 0.5
                continue
            dx_ij = xm[i] - xp[j]
            dy_ij = ym[i] - yp[j]
            # local coordinates
            xi = dx_ij * np.cos(theta[j]) + dy_ij * np.sin(theta[j])
            eta = -dx_ij * np.sin(theta[j]) + dy_ij * np.cos(theta[j])

            r1 = np.sqrt(xi**2 + eta**2)
            r2 = np.sqrt((xi - s[j])**2 + eta**2)

            beta_angle = np.arctan2(eta * s[j], xi * (xi - s[j]) + eta**2)

            # normal velocity influence (vortex)
            u_n = beta_angle / (2 * np.pi)
            # transform back
            cn = (
                np.sin(theta[i] - theta[j]) * (np.log(r2 / r1) / (2 * np.pi))
                + np.cos(theta[i] - theta[j]) * u_n
            )
            A[i, j] = cn

    # Kutta condition: gamma[0] + gamma[n-1] = 0
    A[n, 0] = 1.0
    A[n, n - 1] = 1.0

    # RHS
    b = np.zeros(n + 1)
    for i in range(n):
        b[i] = -np.cos(theta[i] - alpha)
    b[n] = 0.0

    gamma = np.linalg.solve(A, b)

    # tangential velocity and pressure coefficient
    vt = np.zeros(n)
    for i in range(n):
        vt_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            dx_ij = xm[i] - xp[j]
            dy_ij = ym[i] - yp[j]
            xi = dx_ij * np.cos(theta[j]) + dy_ij * np.sin(theta[j])
            eta = -dx_ij * np.sin(theta[j]) + dy_ij * np.cos(theta[j])

            r1 = np.sqrt(xi**2 + eta**2)
            r2 = np.sqrt((xi - s[j])**2 + eta**2)
            beta_angle = np.arctan2(eta * s[j], xi * (xi - s[j]) + eta**2)

            ct = (
                np.cos(theta[i] - theta[j]) * (-np.log(r2 / r1) / (2 * np.pi))
                + np.sin(theta[i] - theta[j]) * beta_angle / (2 * np.pi)
            )
            vt_sum += gamma[j] * ct

        vt[i] = np.sin(theta[i] - alpha) + vt_sum

    cp = 1.0 - vt**2
    cl = sum(gamma[i] * s[i] for i in range(n)) * 2
    return PanelResult(cp, cl, xm, ym, gamma[:n])

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_airfoil_analysis(designation: str, alpha_deg: float = 5.0):
    xp, yp = generate_naca4(designation, n_points=80)
    result = vortex_panel_method(xp, yp, alpha_deg)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Airfoil shape ──
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.fill(xp, yp, color="#4a90d9", alpha=0.3)
    ax.plot(xp, yp, "#4a90d9", linewidth=2)

    # chord line
    ax.plot([0, 1], [0, 0], "--", color="white", alpha=0.3)

    # freestream arrow
    dx_arrow = 0.15 * np.cos(np.radians(alpha_deg))
    dy_arrow = 0.15 * np.sin(np.radians(alpha_deg))
    ax.annotate("", xy=(-0.05 + dx_arrow, dy_arrow), xytext=(-0.05, 0),
                arrowprops=dict(arrowstyle="->", color="#ff6b35", lw=2))
    ax.text(-0.05, -0.06, f"V∞, α={alpha_deg}°", color="#ff6b35", fontsize=10)

    ax.set_aspect("equal")
    ax.set_title(f"NACA {designation} Airfoil — Cₗ = {result.cl:.4f}",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("x/c", color="white")
    ax.set_ylabel("y/c", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, alpha=0.15, color="white")

    # ── Pressure distribution ──
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")

    n_half = len(result.xm) // 2
    ax2.plot(result.xm[:n_half], -result.cp[:n_half], "#ff6b35", linewidth=2, label="Upper surface")
    ax2.plot(result.xm[n_half:], -result.cp[n_half:], "#4a90d9", linewidth=2, label="Lower surface")

    ax2.axhline(0, color="white", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("x/c", color="white")
    ax2.set_ylabel("-Cₚ", color="white")
    ax2.set_title("Pressure Distribution", color="white", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")
    ax2.grid(True, alpha=0.15, color="white")
    ax2.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    return fig


def plot_cl_vs_alpha(designation: str, alpha_range=(-5, 15)):
    alphas = np.linspace(alpha_range[0], alpha_range[1], 21)
    cls = []
    xp, yp = generate_naca4(designation, n_points=80)

    for a in alphas:
        result = vortex_panel_method(xp, yp, a)
        cls.append(result.cl)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(alphas, cls, "#ff6b35", linewidth=2.5, marker="o", markersize=4)
    ax.axhline(0, color="white", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="white", alpha=0.3, linewidth=0.5)

    # linear fit for lift-curve slope
    mask = (alphas >= -3) & (alphas <= 8)
    slope, intercept = np.polyfit(alphas[mask], np.array(cls)[mask], 1)
    alpha_L0 = -intercept / slope
    ax.text(0.02, 0.98,
            f"Lift-curve slope: {slope:.4f} /°\nα_L0 = {alpha_L0:.2f}°\n(2π theory: {2*np.pi*np.pi/180:.4f} /°)",
            transform=ax.transAxes, fontsize=10, verticalalignment="top", color="white",
            bbox=dict(boxstyle="round", facecolor="#2a2a3e", alpha=0.9))

    ax.set_xlabel("Angle of Attack α (°)", color="white")
    ax.set_ylabel("Lift Coefficient Cₗ", color="white")
    ax.set_title(f"NACA {designation} — Cₗ vs α", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, alpha=0.15, color="white")
    plt.tight_layout()
    return fig

# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NACA AIRFOIL ANALYSER")
    print("=" * 60)

    airfoils = ["0012", "2412", "4415"]
    alpha = 5.0

    for naca in airfoils:
        xp, yp = generate_naca4(naca, n_points=80)
        result = vortex_panel_method(xp, yp, alpha)
        print(f"\n  NACA {naca} at α = {alpha}°")
        print(f"    Cₗ = {result.cl:.4f}")
        print(f"    Panels: {len(result.cp)}")

    print("\nGenerating plots...")

    fig1 = plot_airfoil_analysis("2412", alpha_deg=5.0)
    fig1.savefig("naca2412_analysis.png", dpi=150, bbox_inches="tight")
    print("  → naca2412_analysis.png")

    fig2 = plot_cl_vs_alpha("2412")
    fig2.savefig("naca2412_cl_alpha.png", dpi=150, bbox_inches="tight")
    print("  → naca2412_cl_alpha.png")

    fig3 = plot_airfoil_analysis("0012", alpha_deg=8.0)
    fig3.savefig("naca0012_analysis.png", dpi=150, bbox_inches="tight")
    print("  → naca0012_analysis.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
