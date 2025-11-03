import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Physical Properties
# ========================================
# Heating Oil (No. 2 Fuel Oil, typical properties)
rho_oil = 850.0             # kg/m³
cp_oil = 2000.0             # J/(kg·K)
mu_oil = 0.003              # Pa·s (at ~150°C)
k_oil = 0.13                # W/(m·K)
Pr_oil = mu_oil * cp_oil / k_oil  # Prandtl number

# Water - Liquid phase (20-100°C, properties at ~60°C)
rho_water_liq = 988.0       # kg/m³
cp_water_liq = 4180.0       # J/(kg·K)
mu_water_liq = 0.547e-3     # Pa·s
k_water_liq = 0.644         # W/(m·K)
Pr_water_liq = 3.55         # Prandtl number

# Water - Vapor phase (superheated steam, properties at ~150°C)
rho_water_vap = 0.6         # kg/m³ (at 1 atm, ~150°C)
cp_water_vap = 2080.0       # J/(kg·K)
mu_water_vap = 1.5e-5       # Pa·s
k_water_vap = 0.03          # W/(m·K)
Pr_water_vap = mu_water_vap * cp_water_vap / k_water_vap

# Phase change properties
T_sat = 100.0               # Saturation temperature [°C] at 1 atm
h_fg = 2257e3               # Latent heat of vaporization [J/kg]

k_steel = 16.2              # W/(m·K) - Stainless steel 304

# ========================================
# Heat Exchanger Geometry
# ========================================
L = 10.0                    # Length [m]
D_i = 0.02                  # Inner tube diameter [m]
D_o = 0.025                 # Outer tube diameter [m]
D_shell = 0.05              # Shell diameter [m]

# ========================================
# Operating Conditions
# ========================================
# ============================================================
# VARIABLE PARAMETER: Change this to see different results!
# Try: 200°C (no phase change), 250°C (reaches boiling),
#      300°C (full phase change with superheating)
# ============================================================
T_oil_in = 300.0            # Heating oil inlet temperature [°C] - ADJUST THIS!

T_water_in = 20.0           # Water inlet temperature [°C]
m_dot_oil = 0.5             # Oil mass flow rate [kg/s]
m_dot_water = 0.08          # Water mass flow rate [kg/s]

# ========================================
# Functions for Heat Transfer Calculations
# ========================================
def reynolds_number(m_dot, D, mu, rho, A):
    """Calculate Reynolds number"""
    v = m_dot / (rho * A)
    return rho * v * D / mu

def nusselt_dittus_boelter(Re, Pr, heating=True):
    """Dittus-Boelter correlation for turbulent flow"""
    n = 0.4 if heating else 0.3
    if Re > 2300:  # Turbulent flow
        return 0.023 * Re**0.8 * Pr**n
    else:  # Laminar flow (simplified)
        return 3.66

def calculate_h_tube(rho, cp, mu, k, Pr):
    """Calculate tube side (oil) heat transfer coefficient"""
    A_tube = np.pi * D_i**2 / 4
    Re_tube = reynolds_number(m_dot_oil, D_i, mu, rho, A_tube)
    Nu_tube = nusselt_dittus_boelter(Re_tube, Pr, heating=False)
    h_i = Nu_tube * k / D_i
    return h_i, Re_tube

def calculate_h_shell(rho, cp, mu, k, Pr):
    """Calculate shell side (water) heat transfer coefficient"""
    A_shell = np.pi * (D_shell**2 - D_o**2) / 4
    D_h_shell = D_shell - D_o  # Hydraulic diameter approximation
    Re_shell = reynolds_number(m_dot_water, D_h_shell, mu, rho, A_shell)
    Nu_shell = nusselt_dittus_boelter(Re_shell, Pr, heating=True)
    h_o = Nu_shell * k / D_o
    return h_o, Re_shell

def calculate_U(h_i, h_o):
    """Calculate overall heat transfer coefficient"""
    R_conv_i = 1 / h_i
    R_wall = (D_o * np.log(D_o / D_i)) / (2 * k_steel)
    R_conv_o = 1 / h_o
    U = 1 / (R_conv_i + R_wall + R_conv_o)
    return U, R_conv_i, R_wall, R_conv_o

# Calculate heat transfer coefficients for both phases
h_i_oil, Re_tube = calculate_h_tube(rho_oil, cp_oil, mu_oil, k_oil, Pr_oil)
h_o_liq, Re_shell_liq = calculate_h_shell(rho_water_liq, cp_water_liq, mu_water_liq, k_water_liq, Pr_water_liq)
h_o_vap, Re_shell_vap = calculate_h_shell(rho_water_vap, cp_water_vap, mu_water_vap, k_water_vap, Pr_water_vap)

U_liq, R_conv_i, R_wall, R_conv_o_liq = calculate_U(h_i_oil, h_o_liq)
U_vap, _, _, R_conv_o_vap = calculate_U(h_i_oil, h_o_vap)

print("=" * 70)
print("HEAT EXCHANGER PARAMETERS")
print("=" * 70)
print(f"Oil inlet temperature: {T_oil_in:.1f} °C")
print(f"\nTUBE SIDE (Heating Oil):")
print(f"  Reynolds number:         Re = {Re_tube:.0f}")
print(f"  Heat transfer coef.:     h_i = {h_i_oil:.1f} W/(m²·K)")
print(f"\nSHELL SIDE (Water - Liquid Phase):")
print(f"  Reynolds number:         Re = {Re_shell_liq:.0f}")
print(f"  Heat transfer coef.:     h_o = {h_o_liq:.1f} W/(m²·K)")
print(f"  Overall HTC (liquid):    U = {U_liq:.1f} W/(m²·K)")
print(f"\nSHELL SIDE (Water - Vapor Phase):")
print(f"  Reynolds number:         Re = {Re_shell_vap:.0f}")
print(f"  Heat transfer coef.:     h_o = {h_o_vap:.1f} W/(m²·K)")
print(f"  Overall HTC (vapor):     U = {U_vap:.1f} W/(m²·K)")
print("=" * 70 + "\n")

# ========================================
# Numerical Solution with Phase Change
# ========================================
n_nodes = 200               # Number of nodes (increased for better resolution)
dx = L / (n_nodes - 1)      # Spatial step
x = np.linspace(0, L, n_nodes)

# Initialize arrays
T_oil = np.zeros(n_nodes)           # Oil temperature
T_water = np.zeros(n_nodes)         # Water temperature
quality = np.zeros(n_nodes)         # Vapor quality (0=liquid, 1=vapor, 0-1=mixture)
phase = np.zeros(n_nodes, dtype=int) # 0=liquid, 1=sat, 2=vapor

# Boundary conditions (counterflow: oil 0→L, water L→0)
T_oil[0] = T_oil_in
T_water[-1] = T_water_in

# Initial guess for temperature profiles
# For counterflow: oil cools from x=0 to x=L, water heats from x=L to x=0
T_oil_out_guess = max(T_water_in + 30, T_oil_in - 60)
T_water_out_guess = min(T_oil_in - 30, T_sat + 50)
T_oil[:] = np.linspace(T_oil_in, T_oil_out_guess, n_nodes)
T_water[:] = np.linspace(T_water_out_guess, T_water_in, n_nodes)

# Iterative solution with phase change handling
tolerance = 1e-4
max_iterations = 10000
relaxation = 0.6            # Relaxation for stability

for iteration in range(max_iterations):
    T_oil_old = T_oil.copy()
    T_water_old = T_water.copy()

    # Determine water phase at each node based on current temperatures
    for i in range(n_nodes):
        if T_water[i] < T_sat - 1.0:
            phase[i] = 0  # Liquid
            quality[i] = 0.0
        elif T_water[i] > T_sat + 1.0:
            phase[i] = 2  # Vapor
            quality[i] = 1.0
        else:
            phase[i] = 1  # Saturation (phase change zone)
            quality[i] = (T_water[i] - (T_sat - 1.0)) / 2.0

    # Forward integration for oil (x: 0 -> L)
    for i in range(n_nodes - 1):
        # Select U based on average phase between nodes
        if phase[i] >= 1.5:  # Mostly vapor
            U_local = U_vap
        else:  # Liquid or saturation
            U_local = U_liq

        alpha_oil = U_local * np.pi * D_o / (m_dot_oil * cp_oil)
        dT_oil_dx = -alpha_oil * (T_oil[i] - T_water[i])
        T_oil[i+1] = T_oil[i] + dT_oil_dx * dx

    # Backward integration for water (x: L -> 0)
    # Water flows from x=L to x=0, gaining heat from oil
    for i in range(n_nodes - 1, 0, -1):
        # Determine properties based on current phase
        if phase[i] == 2:  # Superheated vapor
            cp_water_local = cp_water_vap
            U_local = U_vap
        elif phase[i] == 1:  # Two-phase region
            # Use a large effective cp to model latent heat absorption
            # The phase change absorbs a lot of energy with little temperature change
            cp_water_local = cp_water_liq * 50.0  # Effective cp during phase change
            U_local = U_liq * 1.5  # Higher heat transfer during boiling
        else:  # Liquid
            cp_water_local = cp_water_liq
            U_local = U_liq

        alpha_water = U_local * np.pi * D_o / (m_dot_water * cp_water_local)
        dT_water_dx = alpha_water * (T_oil[i] - T_water[i])

        # Integrate backwards: T[i-1] = T[i] - dT/dx * dx
        # Since dT/dx > 0 and we're moving backwards (decreasing x),
        # T[i-1] should be less than T[i], meaning water heats up as it flows forward
        T_water[i-1] = T_water[i] - dT_water_dx * dx

        # Ensure water is heating up (outlet > inlet)
        # Since water flows L->0, T[i-1] (closer to outlet) should be > T[i] (closer to inlet)
        # But our equation gives T[i-1] < T[i], which is wrong!
        # FIX: reverse the sign
        T_water[i-1] = T_water[i] + dT_water_dx * dx

    # Apply relaxation for stability
    T_oil = relaxation * T_oil + (1 - relaxation) * T_oil_old
    T_water = relaxation * T_water + (1 - relaxation) * T_water_old

    # Re-enforce boundary conditions after relaxation
    T_oil[0] = T_oil_in
    T_water[-1] = T_water_in

    # Check convergence
    error_oil = np.max(np.abs(T_oil - T_oil_old))
    error_water = np.max(np.abs(T_water - T_water_old))

    if error_oil < tolerance and error_water < tolerance:
        print(f"Converged in {iteration+1} iterations")
        break
else:
    print("Warning: Maximum iterations reached")

# Finalize phase determination
for i in range(n_nodes):
    if T_water[i] < T_sat:
        phase[i] = 0
        quality[i] = 0.0
    elif T_water[i] > T_sat:
        phase[i] = 2
        quality[i] = 1.0
    else:
        phase[i] = 1
        quality[i] = 0.5

# ========================================
# Calculate Heat Transfer with Phase Change
# ========================================
# Heat removed from oil
Q_oil = m_dot_oil * cp_oil * (T_oil[0] - T_oil[-1])

# Heat gained by water (need to account for phase change)
# Water enters at x=L (index -1) and exits at x=0 (index 0)
T_water_in_actual = T_water[-1]
T_water_out_actual = T_water[0]

# Determine phases at inlet and outlet
if T_water_out_actual < T_sat and T_water_in_actual < T_sat:
    # All liquid - no phase change
    Q_sensible_liq = m_dot_water * cp_water_liq * (T_water_out_actual - T_water_in_actual)
    Q_latent = 0.0
    Q_sensible_vap = 0.0
elif T_water_out_actual >= T_sat and T_water_in_actual < T_sat:
    # Phase change occurs: liquid → saturation → vapor
    # Sensible heat to reach saturation
    Q_sensible_liq = m_dot_water * cp_water_liq * (T_sat - T_water_in_actual)
    # Latent heat of vaporization
    Q_latent = m_dot_water * h_fg
    # Sensible heat in vapor phase
    Q_sensible_vap = m_dot_water * cp_water_vap * (T_water_out_actual - T_sat)
elif T_water_out_actual >= T_sat and T_water_in_actual >= T_sat:
    # All vapor
    Q_sensible_liq = 0.0
    Q_latent = 0.0
    Q_sensible_vap = m_dot_water * cp_water_vap * (T_water_out_actual - T_water_in_actual)
else:
    # Edge case
    Q_sensible_liq = 0.0
    Q_latent = 0.0
    Q_sensible_vap = 0.0

Q_water = Q_sensible_liq + Q_latent + Q_sensible_vap

# Find phase change location
if Q_latent > 0:
    phase_change_idx = np.where(np.abs(T_water - T_sat) < 2)[0]
    if len(phase_change_idx) > 0:
        x_phase_change = x[phase_change_idx[0]]
    else:
        x_phase_change = None
else:
    x_phase_change = None

print(f"\nHEAT TRANSFER RESULTS")
print("=" * 70)
print(f"Heat removed from oil:         Q_oil = {Q_oil/1000:.2f} kW")
print(f"\nHeat gained by water:")
print(f"  Sensible heat (liquid):      Q_liq = {Q_sensible_liq/1000:.2f} kW")
print(f"  Latent heat (vaporization):  Q_lat = {Q_latent/1000:.2f} kW")
print(f"  Sensible heat (vapor):       Q_vap = {Q_sensible_vap/1000:.2f} kW")
print(f"  Total heat gained:         Q_water = {Q_water/1000:.2f} kW")
print(f"\nEnergy balance error:                  {abs(Q_oil-Q_water)/Q_oil*100:.2f} %")
print(f"\nOutlet Temperatures:")
print(f"  Oil outlet:    {T_oil[-1]:.2f} °C")
print(f"  Water outlet:  {T_water[0]:.2f} °C")
if x_phase_change is not None:
    print(f"\nPhase change location: x = {x_phase_change:.3f} m (from oil inlet)")
else:
    print(f"\nNo phase change occurred in this configuration")
print("=" * 70 + "\n")

# ========================================
# Plotting Temperature Profiles with Phase Change
# ========================================
fig = plt.figure(figsize=(16, 10))

# Main temperature profile
ax1 = plt.subplot(2, 2, 1)
ax1.plot(x, T_oil, 'r-', linewidth=2.5, label='Heating Oil (tube)', zorder=3)
ax1.plot(x, T_water, 'b-', linewidth=2.5, label='Water (shell)', zorder=3)
ax1.axhline(y=T_sat, color='gray', linestyle='--', linewidth=1.5, label='Saturation temp (100°C)', zorder=1)

# Shade phase regions
liquid_region = T_water < T_sat
vapor_region = T_water > T_sat
if np.any(liquid_region):
    ax1.fill_between(x, 0, 250, where=liquid_region, alpha=0.15, color='blue', label='Liquid region')
if np.any(vapor_region):
    ax1.fill_between(x, 0, 250, where=vapor_region, alpha=0.15, color='red', label='Vapor region')

ax1.set_xlabel('Position along exchanger [m]', fontsize=12)
ax1.set_ylabel('Temperature [°C]', fontsize=12)
ax1.set_title(f'Temperature Profiles (Oil inlet: {T_oil_in}°C)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')
ax1.set_ylim([0, max(T_oil_in + 10, T_water.max() + 10)])

# Temperature difference
ax2 = plt.subplot(2, 2, 2)
Delta_T = T_oil - T_water
ax2.plot(x, Delta_T, 'g-', linewidth=2.5)
ax2.set_xlabel('Position along exchanger [m]', fontsize=12)
ax2.set_ylabel('Temperature Difference ΔT [°C]', fontsize=12)
ax2.set_title('Driving Force (T_oil - T_water)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, Delta_T.max() * 1.2])

# Water phase diagram
ax3 = plt.subplot(2, 2, 3)
phase_colors = ['blue', 'purple', 'red']
phase_names = ['Liquid', 'Saturation', 'Vapor']
for p in range(3):
    mask = phase == p
    if np.any(mask):
        ax3.scatter(x[mask], T_water[mask], c=phase_colors[p], label=phase_names[p],
                   s=20, alpha=0.7)
ax3.axhline(y=T_sat, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Position along exchanger [m]', fontsize=12)
ax3.set_ylabel('Water Temperature [°C]', fontsize=12)
ax3.set_title('Water Phase Distribution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Heat transfer breakdown
ax4 = plt.subplot(2, 2, 4)
heat_components = ['Sensible\n(Liquid)', 'Latent\n(Vaporization)', 'Sensible\n(Vapor)']
heat_values = [Q_sensible_liq/1000, Q_latent/1000, Q_sensible_vap/1000]
colors_bar = ['skyblue', 'orange', 'salmon']
bars = ax4.bar(heat_components, heat_values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Heat Transfer [kW]', fontsize=12)
ax4.set_title('Water Heat Transfer Breakdown', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, val in zip(bars, heat_values):
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} kW', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('heat_exchanger_profiles.png', dpi=300, bbox_inches='tight')
print("Temperature profiles saved to 'heat_exchanger_profiles.png'")
plt.show()
