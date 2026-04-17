function [dxdt, V_term, IR] = cubesat_ode_dch(t, x, p)
% =========================================================================
% cubesat_ode_dch.m  —  v3
%
% ODE for DISCHARGE phase (eclipse)
% State: x = [SOC ; T (K) ; SOH]
%
% v3 change: IR formula updated to include resistance growth with degradation
%   IR = (R0/SOH) * (1 + k_R*(1-SOH)) * Arrhenius(T)
%
%   At SOH=1.00: IR = R0 * Arrhenius                   → 72.934 mΩ at 18.7°C
%   At SOH=0.887: IR = (R0/0.887)*(1+1.899*0.113)*Arr  → 99.806 mΩ at 18.7°C
%
% Thermal model (VACUUM, no convection):
%   dT/dt = (Q_joule + P_heater − Q_rad − Q_cond) / (m·Cp)
% =========================================================================

% --- State bounds ---
SOC = max(p.SOC_min,          min(p.SOC_max,          x(1)));
T   = max(p.T_operating_min,  min(p.T_operating_max,  x(2)));
SOH = max(p.SOH_min,          min(1.05,               x(3)));

% --- Internal resistance (Arrhenius + resistance growth with capacity fade) ---
IR = (p.R0 / SOH) * (1 + p.k_R * (1 - SOH)) * ...
     exp((p.Ea / p.R_gaz) * (1/T - 1/p.T_ref));

% --- SOC stress factor (U-shaped) ---
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% --- dSOC/dt ---
dSOC = -(p.eta * p.I_dch) / (p.Q_nom * SOH * 3600);

% --- dT/dt : vacuum thermal balance ---
T_space = 3.0;
Q_rad   = p.sigma * p.emissivity * p.A_rad * (T^4 - T_space^4);
Q_cond  = p.G_structure * (T - p.T_amb_eclipse);
Q_joule = p.I_dch^2 * IR;
P_heat  = p.P_heater * double(T < p.T_heater_on);
dT      = (Q_joule + P_heat - Q_rad - Q_cond) / (p.m * p.Cp);

% --- dSOH/dt ---
dSOH = -p.alpha * p.I_dch * exp(-p.Ea / (p.R_gaz * T)) * stress * p.gamma;

dxdt = [dSOC; dT; dSOH];

% --- Terminal voltage ---
V_term = p.OCV(SOC) - IR * p.I_dch;
V_term = max(p.V_pack_min, min(p.V_pack_max, V_term));

end
