import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Physical Properties (Water at ~50°C and Stainless Steel 304)
# ========================================
rho_water = 988.0           # kg/m³
cp_water = 4180.0           # J/(kg·K)
mu_water = 0.547e-3         # Pa·s
k_water = 0.644             # W/(m·K)
Pr_water = 3.55             # Prandtl number

k_steel = 16.2              # W/(m·K) - Stainless steel 304

# ========================================
# Heat Exchanger Geometry
# ========================================
L = 2.0                     # Length [m]
D_i = 0.02                  # Inner tube diameter [m]
D_o = 0.025                 # Outer tube diameter [m]
D_shell = 0.05              # Shell diameter [m]

# ========================================
# Operating Conditions
# ========================================
T_h_in = 80.0               # Hot fluid inlet temperature [°C]
T_c_in = 20.0               # Cold fluid inlet temperature [°C]
m_dot_h = 0.5               # Hot fluid mass flow rate [kg/s]
m_dot_c = 0.5               # Cold fluid mass flow rate [kg/s]

# ========================================
# Calculate Convection Coefficients (Dittus-Boelter)
# ========================================
def reynolds_number(m_dot, D, mu, A):
    """Calculate Reynolds number"""
    v = m_dot / (rho_water * A)
    return rho_water * v * D / mu

def nusselt_dittus_boelter(Re, Pr, heating=True):
    """Dittus-Boelter correlation for turbulent flow"""
    n = 0.4 if heating else 0.3
    return 0.023 * Re**0.8 * Pr**n

# Tube side (hot fluid)
A_tube = np.pi * D_i**2 / 4
Re_tube = reynolds_number(m_dot_h, D_i, mu_water, A_tube)
Nu_tube = nusselt_dittus_boelter(Re_tube, Pr_water, heating=False)
h_i = Nu_tube * k_water / D_i

# Shell side (cold fluid) - simplified as flow over cylinder
A_shell = np.pi * (D_shell**2 - D_o**2) / 4
D_h_shell = D_shell - D_o  # Hydraulic diameter approximation
Re_shell = reynolds_number(m_dot_c, D_h_shell, mu_water, A_shell)
Nu_shell = nusselt_dittus_boelter(Re_shell, Pr_water, heating=True)
h_o = Nu_shell * k_water / D_o

# Overall heat transfer coefficient
R_conv_i = 1 / h_i
R_wall = (D_o * np.log(D_o / D_i)) / (2 * k_steel)
R_conv_o = 1 / h_o
U = 1 / (R_conv_i + R_wall + R_conv_o)

print("=" * 60)
print("HEAT EXCHANGER PARAMETERS")
print("=" * 60)
print(f"Reynolds number (tube):  {Re_tube:.0f}")
print(f"Reynolds number (shell): {Re_shell:.0f}")
print(f"Heat transfer coef. (tube side):  h_i = {h_i:.1f} W/(m²·K)")
print(f"Heat transfer coef. (shell side): h_o = {h_o:.1f} W/(m²·K)")
print(f"Overall heat transfer coef.:      U   = {U:.1f} W/(m²·K)")
print(f"Thermal resistances:")
print(f"  R_conv_inner = {R_conv_i:.6f} (m²·K)/W")
print(f"  R_wall       = {R_wall:.6f} (m²·K)/W")
print(f"  R_conv_outer = {R_conv_o:.6f} (m²·K)/W")
print("=" * 60 + "\n")

# ========================================
# Numerical Solution using Finite Difference Method
# ========================================
n_nodes = 100               # Number of nodes
dx = L / (n_nodes - 1)      # Spatial step
x = np.linspace(0, L, n_nodes)

# Initialize temperature arrays
T_h = np.zeros(n_nodes)     # Hot fluid temperature
T_c = np.zeros(n_nodes)     # Cold fluid temperature

# Boundary conditions
T_h[0] = T_h_in             # Hot fluid enters at x=0
T_c[-1] = T_c_in            # Cold fluid enters at x=L (counterflow)

# Coefficients for the differential equations
alpha_h = U * np.pi * D_o / (m_dot_h * cp_water)
alpha_c = U * np.pi * D_o / (m_dot_c * cp_water)

# Iterative solution (shooting method with relaxation)
tolerance = 1e-6
max_iterations = 10000
relaxation = 0.5            # Relaxation factor for stability

for iteration in range(max_iterations):
    T_h_old = T_h.copy()
    T_c_old = T_c.copy()
    
    # Forward integration for hot fluid (x: 0 -> L)
    for i in range(n_nodes - 1):
        dT_h_dx = -alpha_h * (T_h[i] - T_c[i])
        T_h[i+1] = T_h[i] + dT_h_dx * dx
    
    # Backward integration for cold fluid (x: L -> 0)
    for i in range(n_nodes - 1, 0, -1):
        dT_c_dx = alpha_c * (T_h[i] - T_c[i])
        T_c[i-1] = T_c[i] - dT_c_dx * dx
    
    # Apply relaxation for stability
    T_h = relaxation * T_h + (1 - relaxation) * T_h_old
    T_c = relaxation * T_c + (1 - relaxation) * T_c_old
    
    # Check convergence
    error_h = np.max(np.abs(T_h - T_h_old))
    error_c = np.max(np.abs(T_c - T_c_old))
    
    if error_h < tolerance and error_c < tolerance:
        print(f"Converged in {iteration+1} iterations")
        break
else:
    print("Warning: Maximum iterations reached")

# ========================================
# Calculate Heat Transfer
# ========================================
Q_h = m_dot_h * cp_water * (T_h[0] - T_h[-1])
Q_c = m_dot_c * cp_water * (T_c[-1] - T_c[0])
Q_avg = (Q_h + Q_c) / 2

print(f"\nHEAT TRANSFER RESULTS")
print("=" * 60)
print(f"Heat removed from hot fluid: Q_h = {Q_h/1000:.2f} kW")
print(f"Heat gained by cold fluid:   Q_c = {Q_c/1000:.2f} kW")
print(f"Average heat transfer:       Q   = {Q_avg/1000:.2f} kW")
print(f"Energy balance error:              {abs(Q_h-Q_c)/Q_avg*100:.2f} %")
print(f"\nOutlet Temperatures:")
print(f"Hot fluid outlet:  {T_h[-1]:.2f} °C")
print(f"Cold fluid outlet: {T_c[0]:.2f} °C")
print("=" * 60 + "\n")

# ========================================
# Plotting Temperature Profiles
# ========================================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x, T_h, 'r-', linewidth=2, label='Hot fluid (tube)')
plt.plot(x, T_c, 'b-', linewidth=2, label='Cold fluid (shell)')
plt.xlabel('Length x [m]', fontsize=12)
plt.ylabel('Temperature [°C]', fontsize=12)
plt.title('Temperature Profiles along Heat Exchanger', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.subplot(1, 2, 2)
Delta_T = T_h - T_c
plt.plot(x, Delta_T, 'g-', linewidth=2)
plt.xlabel('Length x [m]', fontsize=12)
plt.ylabel('Temperature Difference ΔT [°C]', fontsize=12)
plt.title('Temperature Difference (T_hot - T_cold)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/heat_exchanger_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

print("Temperature profiles saved to 'heat_exchanger_profiles.png'")
