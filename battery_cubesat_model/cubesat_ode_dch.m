function [dxdt, V_term, IR] = cubesat_ode_dch(t, x, p)
% =========================================================================
% ODE for DISCHARGE phase (eclipse in LEO)
% x = [SOC; T(K); SOH]
% I_dch is positive → SOC decreases
% 
% CORRECTED: 
%  - Uses radiation/conduction in vacuum (not convection)
%  - Keeps temperature within managed operating range with heater logic
%  - Bounds all states strictly
% =========================================================================

% --- State extraction and bounds enforcement ---
SOC = max(p.SOC_min, min(p.SOC_max, x(1)));
T = max(p.T_operating_min, min(p.T_operating_max, x(2)));  % Keep in [-10, +50]°C band
SOH = max(p.SOH_min, min(1.05, x(3)));

% --- Internal resistance (Arrhenius model) ---
IR = (p.R0 / SOH) * exp((p.Ea / p.R_gaz) * (1/T - 1/p.T_ref));

% --- SOC stress factor (U-shaped, higher stress at extremes) ---
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% --- dSOC/dt: discharge decreases SOC ---
dSOC = -(p.eta * p.I_dch) / (p.Q_nom * SOH * 3600);

% --- dT/dt: CORRECTED THERMAL MODEL FOR VACUUM ---
% Energy balance in vacuum:
%   Q_Joule = I²·R (Joule heating from discharge)
%   Q_rad = σ·ε·A·(T⁴ - T_space⁴) (radiation to deep space, ~3K)
%   Q_cond = G_structure·(T - T_structure) (conduction to spacecraft structure)
%   P_heater = heater power (if T < T_heater_on)
%   dT/dt = (Q_Joule + P_heater - Q_rad - Q_cond) / (m·Cp)

% During eclipse, ambient is ~263K (-10°C), structure cools slowly
T_space = 3.0;                          % Deep space radiation sink (K)
T_structure = p.T_amb_eclipse;          % Structure temperature (simplification: follows ambient)

% Radiation heat loss (W)
Q_rad = p.sigma * p.emissivity * p.A_rad * (T^4 - T_space^4);

% Conduction to structure (W)
Q_cond = p.G_structure * (T - T_structure);

% Joule heating from discharge (W)
Q_joule = p.I_dch^2 * IR;

% Heater control (simple thermostat)
if T < p.T_heater_on
    P_heater = p.P_heater;              % Turn on heater
elseif T > p.T_heater_off
    P_heater = 0.0;                     % Turn off heater
else
    P_heater = 0.0;                     % Hysteresis: stay off until T_heater_on
end

% Temperature derivative
dT = (Q_joule + P_heater - Q_rad - Q_cond) / (p.m * p.Cp);

% --- dSOH/dt: aging from current, temperature, stress ---
% Higher temperature and current → exponentially higher aging
dSOH = -p.alpha * p.I_dch * exp(-p.Ea / (p.R_gaz * T)) * stress * p.gamma;

dxdt = [dSOC; dT; dSOH];

% --- Terminal voltage ---
V_term = p.OCV(SOC) - IR * p.I_dch;
V_term = max(p.V_pack_min, min(p.V_pack_max, V_term));  % Clamp only as hard limit

end
