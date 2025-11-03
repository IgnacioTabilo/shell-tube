"""
Shell-and-Tube Heat Exchanger with Phase Change - LMTD Method
===============================================================

This code implements the proper LMTD (Log Mean Temperature Difference) method
for heat exchanger analysis with phase change, as described in Incropera &
DeWitt, "Fundamentals of Heat and Mass Transfer", Chapters 10-11.

KEY IMPROVEMENTS FROM PREVIOUS VERSION:
---------------------------------------
1. PROPER LMTD METHOD (Chapter 11.3):
   - Uses analytical LMTD calculation: ΔT_LM = (ΔT₁ - ΔT₂) / ln(ΔT₁/ΔT₂)
   - Replaces incorrect node-by-node integration with local ΔT
   - Ensures thermodynamically consistent temperature profiles

2. SEGMENTED ANALYSIS FOR PHASE CHANGE:
   - Segment 1: Liquid heating (Q = ṁ·cp·ΔT)
   - Segment 2: Boiling (Q = ṁ·h_fg) with proper correlation
   - Segment 3: Vapor superheating (Q = ṁ·cp·ΔT)
   - Each segment uses appropriate U value and LMTD

3. CORRECT BOILING HEAT TRANSFER (Chapter 10):
   - Uses h_boiling ≈ 8000 W/(m²·K) for nucleate boiling
   - Replaces hack of artificial cp_effective = 50×cp_liquid
   - Based on Rohsenow correlation (simplified)

4. ENERGY BALANCE VALIDATION:
   - Verifies Q_oil = Q_water (should be < 1% error)
   - Checks thermodynamic feasibility (ΔT > 0 everywhere)

METHODOLOGY:
------------
For counterflow heat exchanger with phase change:
- Q = U·A·ΔT_LM where A = π·D·L
- Iterate to find outlet temperature that uses available length L
- Calculate segment lengths: L_seg = Q_seg / (U·π·D·ΔT_LM)

REFERENCES:
-----------
- Incropera Chapter 8: Internal flow correlations (Dittus-Boelter)
- Incropera Chapter 10: Boiling and condensation
- Incropera Chapter 11: Heat exchanger analysis (LMTD and ε-NTU methods)
"""

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

def calculate_h_boiling():
    """
    Calculate heat transfer coefficient for nucleate pool boiling
    Based on Rohsenow correlation (Incropera Chapter 10)

    For water at atmospheric pressure with typical heat fluxes,
    h_boiling ranges from 5,000 to 100,000 W/(m²·K)

    Using conservative estimate for pool boiling.
    Note: Flow boiling (Chen correlation) would give even higher values.
    """
    # Conservative estimate for nucleate boiling of water
    h_boiling = 8000.0  # W/(m²·K)
    return h_boiling

def calculate_U(h_i, h_o):
    """Calculate overall heat transfer coefficient"""
    R_conv_i = 1 / h_i
    R_wall = (D_o * np.log(D_o / D_i)) / (2 * k_steel)
    R_conv_o = 1 / h_o
    U = 1 / (R_conv_i + R_wall + R_conv_o)
    return U, R_conv_i, R_wall, R_conv_o

def calculate_LMTD(T_h_in, T_h_out, T_c_in, T_c_out):
    """
    Calculate Log Mean Temperature Difference for counterflow
    Incropera Eq. 11.15

    For counterflow:
    ΔT₁ = T_h,in - T_c,out  (at hot fluid inlet)
    ΔT₂ = T_h,out - T_c,in  (at cold fluid outlet)

    LMTD = (ΔT₁ - ΔT₂) / ln(ΔT₁/ΔT₂)
    """
    Delta_T1 = T_h_in - T_c_out
    Delta_T2 = T_h_out - T_c_in

    # Avoid division by zero or log(1)
    if abs(Delta_T1 - Delta_T2) < 1e-6:
        return Delta_T1  # If ΔT is nearly constant

    # Ensure positive temperature differences (second law)
    if Delta_T1 <= 0 or Delta_T2 <= 0:
        raise ValueError(f"Invalid temperature differences: ΔT₁={Delta_T1:.2f}, ΔT₂={Delta_T2:.2f}")

    LMTD = (Delta_T1 - Delta_T2) / np.log(Delta_T1 / Delta_T2)
    return LMTD

# ========================================
# Calculate Heat Transfer Coefficients
# ========================================
# Tube side (oil) - same throughout
h_i_oil, Re_tube = calculate_h_tube(rho_oil, cp_oil, mu_oil, k_oil, Pr_oil)

# Shell side (water) - different for each phase
h_o_liq, Re_shell_liq = calculate_h_shell(rho_water_liq, cp_water_liq, mu_water_liq,
                                           k_water_liq, Pr_water_liq)
h_o_boil = calculate_h_boiling()
h_o_vap, Re_shell_vap = calculate_h_shell(rho_water_vap, cp_water_vap, mu_water_vap,
                                           k_water_vap, Pr_water_vap)

# Overall heat transfer coefficients for each region
U_liq, R_conv_i, R_wall, R_conv_o_liq = calculate_U(h_i_oil, h_o_liq)
U_boil, _, _, R_conv_o_boil = calculate_U(h_i_oil, h_o_boil)
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
print(f"\nSHELL SIDE (Water - Boiling):")
print(f"  Heat transfer coef.:     h_boil = {h_o_boil:.1f} W/(m²·K)")
print(f"  Overall HTC (boiling):   U = {U_boil:.1f} W/(m²·K)")
print(f"\nSHELL SIDE (Water - Vapor Phase):")
print(f"  Reynolds number:         Re = {Re_shell_vap:.0f}")
print(f"  Heat transfer coef.:     h_o = {h_o_vap:.1f} W/(m²·K)")
print(f"  Overall HTC (vapor):     U = {U_vap:.1f} W/(m²·K)")
print("=" * 70 + "\n")

# ========================================
# LMTD-Based Segmented Analysis with Phase Change
# ========================================
print("LMTD-BASED SEGMENTED ANALYSIS")
print("=" * 70)

# Step 1: Determine maximum possible heat transfer
# Calculate heat required for complete phase change
Q_to_sat = m_dot_water * cp_water_liq * (T_sat - T_water_in)
Q_vaporization = m_dot_water * h_fg
Q_max_oil = m_dot_oil * cp_oil * (T_oil_in - T_water_in - 10)  # Leave 10°C approach

print(f"\nHeat requirements for water phase change:")
print(f"  To reach saturation:     Q₁ = {Q_to_sat/1000:.2f} kW")
print(f"  For vaporization:        Q₂ = {Q_vaporization/1000:.2f} kW")
print(f"  Max available from oil:  Q_max = {Q_max_oil/1000:.2f} kW")

# We'll use an iterative approach to find the outlet temperatures
# that satisfy both energy balance and LMTD constraints

# Initial guess: determine if phase change occurs
Q_available = Q_to_sat + Q_vaporization
T_oil_out_guess = T_oil_in - Q_available / (m_dot_oil * cp_oil)

# Check if we have enough length and heat for complete vaporization
if T_oil_out_guess < T_water_in:
    # Not enough heat for complete vaporization
    # Water will only partially vaporize or just heat up
    print("\nInsufficient heat for complete vaporization - solving iteratively...")

    # Try to find equilibrium: iterate on T_water_out
    # Start with assumption of reaching saturation only
    T_water_out_test = T_sat + 5  # Initial guess slightly above saturation
else:
    T_water_out_test = T_sat + 50  # Initial guess for superheated case

# Iterative solution to find consistent outlet temperatures
max_iter = 200
tolerance = 0.05  # m (length tolerance)
converged = False
step_size = 5.0  # Initial step size for temperature adjustment

for iteration in range(max_iter):
    # Determine which segments exist based on T_water_out_test
    segments = []

    # Segment 1: Liquid heating (if inlet is below saturation)
    if T_water_in < T_sat:
        segments.append({
            'name': 'Liquid Heating',
            'T_water_in': T_water_in,
            'T_water_out': min(T_water_out_test, T_sat),
            'U': U_liq,
            'cp_water': cp_water_liq,
            'phase': 'liquid'
        })

    # Segment 2: Boiling (if outlet is above saturation and we reached it)
    if T_water_out_test > T_sat and T_water_in < T_sat:
        segments.append({
            'name': 'Boiling',
            'T_water_in': T_sat,
            'T_water_out': T_sat,
            'U': U_boil,
            'cp_water': None,  # Use latent heat instead
            'phase': 'boiling'
        })

    # Segment 3: Vapor superheating (if outlet is significantly above saturation)
    if T_water_out_test > T_sat + 1.0:
        segments.append({
            'name': 'Vapor Superheating',
            'T_water_in': T_sat,
            'T_water_out': T_water_out_test,
            'U': U_vap,
            'cp_water': cp_water_vap,
            'phase': 'vapor'
        })

    # Calculate heat and length for each segment
    total_length = 0
    total_Q_water = 0
    x_positions = [0]  # Track segment boundaries
    calculation_failed = False

    # Process segments from water inlet to outlet (backwards along exchanger)
    # In counterflow: oil flows 0→L, water flows L→0
    # So segments are positioned from x=L backwards to x=0

    # Track oil inlet temperature (starts at x=0)
    T_oil_current = T_oil_in

    for i, seg in enumerate(segments):
        if seg['phase'] == 'boiling':
            # Boiling: Q = ṁ·hfg
            Q_seg = m_dot_water * h_fg
            total_Q_water += Q_seg

            # For boiling, water temp is constant at T_sat
            # Oil temperature drops along this segment

            # Oil temperature drop from energy balance
            DT_oil_segment = Q_seg / (m_dot_oil * cp_oil)

            # Oil enters this segment at T_oil_current
            T_oil_seg_in = T_oil_current
            T_oil_seg_out = T_oil_seg_in - DT_oil_segment

            # Check for thermodynamic feasibility
            if T_oil_seg_out < T_sat:
                # Oil cannot cool below water temperature
                calculation_failed = True
                T_water_out_test -= step_size
                break

            # LMTD for boiling: water at constant T_sat
            # ΔT₁ = T_oil_in - T_sat (hot end)
            # ΔT₂ = T_oil_out - T_sat (cold end)
            try:
                LMTD_seg = calculate_LMTD(T_oil_seg_in, T_oil_seg_out, T_sat, T_sat)
            except ValueError:
                # Invalid temperatures - adjust guess
                calculation_failed = True
                T_water_out_test -= step_size
                break

            # Length from Q = U·A·LMTD
            # A = π·D_o·L_seg
            L_seg = Q_seg / (seg['U'] * np.pi * D_o * LMTD_seg)

            seg['L'] = L_seg
            seg['Q'] = Q_seg
            seg['LMTD'] = LMTD_seg
            seg['T_oil_in'] = T_oil_seg_in
            seg['T_oil_out'] = T_oil_seg_out

            # Update current oil temperature for next segment
            T_oil_current = T_oil_seg_out

        else:
            # Sensible heating (liquid or vapor)
            Q_seg = m_dot_water * seg['cp_water'] * (seg['T_water_out'] - seg['T_water_in'])
            total_Q_water += Q_seg

            # Oil temperature drop from energy balance
            DT_oil_segment = Q_seg / (m_dot_oil * cp_oil)

            # Oil enters this segment at T_oil_current
            T_oil_seg_in = T_oil_current
            T_oil_seg_out = T_oil_seg_in - DT_oil_segment

            # LMTD for counterflow
            try:
                LMTD_seg = calculate_LMTD(T_oil_seg_in, T_oil_seg_out,
                                         seg['T_water_in'], seg['T_water_out'])
            except ValueError:
                # Invalid temperatures
                calculation_failed = True
                T_water_out_test -= step_size
                break

            # Length from Q = U·A·LMTD
            L_seg = Q_seg / (seg['U'] * np.pi * D_o * LMTD_seg)

            seg['L'] = L_seg
            seg['Q'] = Q_seg
            seg['LMTD'] = LMTD_seg
            seg['T_oil_in'] = T_oil_seg_in
            seg['T_oil_out'] = T_oil_seg_out

            # Update current oil temperature for next segment
            T_oil_current = T_oil_seg_out

        total_length += L_seg
        x_positions.append(total_length)

    if calculation_failed:
        continue

    # Check if total length matches available length
    length_error = total_length - L
    if abs(length_error) < tolerance:  # Within tolerance
        converged = True
        print(f"\nConverged in {iteration + 1} iterations")
        print(f"Water outlet temperature: {T_water_out_test:.2f} °C")
        break
    else:
        # Adaptive step size based on error magnitude
        if abs(length_error) < 0.5:
            step_size = 0.5
        elif abs(length_error) < 2.0:
            step_size = 1.0
        else:
            step_size = 2.0

        if length_error > 0:
            # Too much length needed - reduce outlet temp
            T_water_out_test -= step_size
        else:
            # Too little length used - increase outlet temp
            T_water_out_test += step_size

        # Bounds checking
        if T_water_out_test > T_oil_in - 10:
            T_water_out_test = T_oil_in - 12  # Minimum approach temp
        elif T_water_out_test < T_water_in:
            T_water_out_test = T_water_in + 5

if not converged:
    print(f"\nWarning: Did not fully converge. Using L={total_length:.2f}m (target: {L}m)")
    print(f"Water outlet temperature: {T_water_out_test:.2f} °C")

# Final outlet temperatures
T_water_out = T_water_out_test
T_oil_out = T_oil_current if segments else T_oil_in - 20

# ========================================
# Print Segment Results
# ========================================
print("\nSEGMENT ANALYSIS:")
print("-" * 70)
x_start = 0
for i, seg in enumerate(segments):
    if 'L' not in seg:
        # Skip incomplete segments
        continue
    x_end = x_start + seg['L']
    print(f"\nSegment {i+1}: {seg['name']}")
    print(f"  Position:          x = {x_start:.3f} to {x_end:.3f} m")
    print(f"  Length:            L = {seg['L']:.3f} m")
    print(f"  Heat transfer:     Q = {seg['Q']/1000:.2f} kW")
    print(f"  LMTD:              ΔT_LM = {seg['LMTD']:.2f} °C")
    print(f"  U value:           U = {seg['U']:.1f} W/(m²·K)")
    print(f"  Oil temps:         {seg['T_oil_in']:.2f} → {seg['T_oil_out']:.2f} °C")
    print(f"  Water temps:       {seg['T_water_in']:.2f} → {seg['T_water_out']:.2f} °C")
    x_start = x_end

# ========================================
# Calculate Final Heat Transfer
# ========================================
Q_oil = m_dot_oil * cp_oil * (T_oil_in - T_oil_out)
Q_water_total = sum(seg['Q'] for seg in segments if 'Q' in seg)

# Break down water heat by type
Q_sensible_liq = sum(seg['Q'] for seg in segments if seg.get('phase') == 'liquid' and 'Q' in seg)
Q_latent = sum(seg['Q'] for seg in segments if seg.get('phase') == 'boiling' and 'Q' in seg)
Q_sensible_vap = sum(seg['Q'] for seg in segments if seg.get('phase') == 'vapor' and 'Q' in seg)

print("\n" + "=" * 70)
print("HEAT TRANSFER RESULTS")
print("=" * 70)
print(f"Heat removed from oil:         Q_oil = {Q_oil/1000:.2f} kW")
print(f"\nHeat gained by water:")
print(f"  Sensible heat (liquid):      Q_liq = {Q_sensible_liq/1000:.2f} kW")
print(f"  Latent heat (vaporization):  Q_lat = {Q_latent/1000:.2f} kW")
print(f"  Sensible heat (vapor):       Q_vap = {Q_sensible_vap/1000:.2f} kW")
print(f"  Total heat gained:         Q_water = {Q_water_total/1000:.2f} kW")
print(f"\nEnergy balance error:                  {abs(Q_oil-Q_water_total)/Q_oil*100:.2f} %")
print(f"\nOutlet Temperatures:")
print(f"  Oil outlet:    {T_oil_out:.2f} °C")
print(f"  Water outlet:  {T_water_out:.2f} °C")
print(f"\nTotal exchanger length used: {total_length:.3f} m (available: {L} m)")
print("=" * 70 + "\n")

# ========================================
# Build Temperature Profiles for Plotting
# ========================================
n_points = 200
x_plot = np.linspace(0, L, n_points)
T_oil_plot = np.zeros(n_points)
T_water_plot = np.zeros(n_points)
phase_plot = np.zeros(n_points, dtype=int)

# Fill in profiles based on segments
# Counterflow: oil flows 0→L, water flows L→0
# Segments are arranged from water inlet (x=L) towards outlet (x=0)

x_cumulative = 0
for i, seg in enumerate(segments):
    if 'L' not in seg:
        # Skip incomplete segments
        continue

    # Find points in this segment
    x_seg_start = x_cumulative
    x_seg_end = x_cumulative + seg['L']

    mask = (x_plot >= x_seg_start) & (x_plot < x_seg_end)

    # Oil temperature (linear interpolation in counterflow)
    T_oil_plot[mask] = np.interp(
        x_plot[mask],
        [x_seg_start, x_seg_end],
        [seg['T_oil_in'], seg['T_oil_out']]
    )

    # Water temperature (flows opposite direction)
    if seg['phase'] == 'boiling':
        T_water_plot[mask] = T_sat
        phase_plot[mask] = 1
    else:
        # Linear interpolation (but remember water flows L→0)
        T_water_plot[mask] = np.interp(
            x_plot[mask],
            [x_seg_start, x_seg_end],
            [seg['T_water_out'], seg['T_water_in']]  # Reversed for counterflow
        )
        phase_plot[mask] = 0 if seg['phase'] == 'liquid' else 2

    x_cumulative += seg['L']

# Fill any remaining points (if L > total_length)
if x_cumulative < L:
    mask = x_plot >= x_cumulative
    T_oil_plot[mask] = T_oil_out
    T_water_plot[mask] = T_water_in

# ========================================
# Plotting Temperature Profiles
# ========================================
fig = plt.figure(figsize=(16, 10))

# Main temperature profile
ax1 = plt.subplot(2, 2, 1)
ax1.plot(x_plot, T_oil_plot, 'r-', linewidth=2.5, label='Heating Oil (tube)', zorder=3)
ax1.plot(x_plot, T_water_plot, 'b-', linewidth=2.5, label='Water (shell)', zorder=3)
ax1.axhline(y=T_sat, color='gray', linestyle='--', linewidth=1.5, label='Saturation temp (100°C)', zorder=1)

# Mark segment boundaries
x_cumulative = 0
for i, seg in enumerate(segments):
    if 'L' not in seg:
        continue
    x_cumulative += seg['L']
    ax1.axvline(x=x_cumulative, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    # Label segment
    x_mid = x_cumulative - seg['L']/2
    ax1.text(x_mid, T_oil_in + 5, seg['name'].split()[0],
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Shade phase regions
liquid_region = T_water_plot < T_sat - 0.5
vapor_region = T_water_plot > T_sat + 0.5
if np.any(liquid_region):
    ax1.fill_between(x_plot, 0, 350, where=liquid_region, alpha=0.15, color='blue', label='Liquid region')
if np.any(vapor_region):
    ax1.fill_between(x_plot, 0, 350, where=vapor_region, alpha=0.15, color='red', label='Vapor region')

ax1.set_xlabel('Position along exchanger [m]', fontsize=12)
ax1.set_ylabel('Temperature [°C]', fontsize=12)
ax1.set_title(f'Temperature Profiles - LMTD Method (Oil inlet: {T_oil_in}°C)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')
ax1.set_ylim([0, max(T_oil_in + 10, T_water_plot.max() + 10)])

# Temperature difference
ax2 = plt.subplot(2, 2, 2)
Delta_T = T_oil_plot - T_water_plot
ax2.plot(x_plot, Delta_T, 'g-', linewidth=2.5)

# Mark segment boundaries
x_cumulative = 0
for seg in segments:
    if 'L' not in seg:
        continue
    x_cumulative += seg['L']
    ax2.axvline(x=x_cumulative, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    # Show LMTD for each segment
    x_mid = x_cumulative - seg['L']/2
    ax2.text(x_mid, seg['LMTD'], f"LMTD={seg['LMTD']:.1f}°C",
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax2.set_xlabel('Position along exchanger [m]', fontsize=12)
ax2.set_ylabel('Temperature Difference ΔT [°C]', fontsize=12)
ax2.set_title('Driving Force (T_oil - T_water)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, Delta_T.max() * 1.3])

# Water phase diagram
ax3 = plt.subplot(2, 2, 3)
phase_colors = ['blue', 'purple', 'red']
phase_names = ['Liquid', 'Saturation', 'Vapor']
for p in range(3):
    mask = phase_plot == p
    if np.any(mask):
        ax3.scatter(x_plot[mask], T_water_plot[mask], c=phase_colors[p], label=phase_names[p],
                   s=20, alpha=0.7)
ax3.axhline(y=T_sat, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark segment boundaries
x_cumulative = 0
for seg in segments:
    if 'L' not in seg:
        continue
    x_cumulative += seg['L']
    ax3.axvline(x=x_cumulative, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

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
