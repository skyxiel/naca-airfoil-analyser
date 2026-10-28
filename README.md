# NACA Airfoil Analyser

Generates NACA 4-digit airfoil geometry and solves for pressure distribution and lift coefficient using a vortex panel method — inviscid aerodynamics from scratch, no external solver.

![NACA 2412 analysis](naca2412_analysis.png)
![Cl vs alpha](naca2412_cl_alpha.png)

## What it does

| Feature | Detail |
|---|---|
| **Airfoil generation** | Any NACA 4-digit profile (symmetric or cambered) with cosine-spaced panels for sharp leading-edge resolution |
| **Vortex panel method** | Builds and solves the influence-coefficient matrix for N panels, enforces the Kutta condition at the trailing edge |
| **Pressure distribution** | Computes Cₚ on upper and lower surfaces from panel tangential velocities |
| **Lift coefficient** | Integrates panel circulation to get Cₗ — validated against thin-airfoil theory (2π per radian) |
| **Lift curve** | Sweeps angle of attack, fits the lift-curve slope, finds zero-lift angle |
| **Visualisation** | Airfoil shape, pressure distribution, and Cₗ-vs-α plots |

## Example output

```
  NACA 0012 at α = 5.0°
    Cₗ = 0.5553
    Panels: 158

  NACA 2412 at α = 5.0°
    Cₗ = 0.7945
    Panels: 158

  NACA 4415 at α = 5.0°
    Cₗ = 1.0298
    Panels: 158
```

## Usage

```bash
pip install -r requirements.txt
python airfoil_analyser.py
```

## Built with

- Python 3.10+
- NumPy
- Matplotlib

## What I learned

How panel methods work — discretising the airfoil surface into panels, computing velocity influence coefficients, enforcing the Kutta condition so the flow leaves the trailing edge smoothly. Why cosine spacing matters at the leading edge where curvature is highest. How cambered airfoils generate lift at zero angle of attack, and why the lift-curve slope converges to 2π/rad for thin airfoils.
