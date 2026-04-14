function [dxdt, V_term, IR] = cubesat_ode_ch(t, x, p)

% --- State extraction and bounds ---
SOC = max(p.SOC_min, min(p.SOC_max, x(1)));
T = max(p.T_operating_min, min(p.T_operating_max, x(2)));
SOH = max(p.SOH_min, min(1.05, x(3)));

% --- Internal resistance (Arrhenius) ---
IR = (p.R0 / SOH) * exp((p.Ea / p.R_gaz) * (1/T - 1/p.T_ref));

% --- SOC stress factor (U-shaped) ---
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% --- dSOC/dt: charging increases SOC ---
% Charge current is input; accepted charge follows battery acceptance curve
dSOC = (p.eta * p.I_ch) / (p.Q_nom * SOH * 3600);

% --- dT/dt: CORRECTED THERMAL MODEL FOR VACUUM ---
% During sunlight, solar panel heats the battery structure
% T_structure during sunlight ≈ T_amb_sun ≈ 323K (+50°C hotspot)
%
% Energy balance:
%   Q_Joule = I²·R (resistive heating during charge)
%   Q_solar ≈ absorbed solar energy coupled through structure
%   Q_rad = σ·ε·A·(T⁴ - T_space⁴) (radiation to space)
%   Q_cond = G_structure·(T - T_structure) (conduction from structure)
%   dT/dt = (Q_Joule + Q_cond - Q_rad) / (m·Cp)

T_space = 3.0;                          % Deep space sink (K)
T_structure = p.T_amb_sun;              % Structure temperature during sunlight

% Radiation (same as discharge)
Q_rad = p.sigma * p.emissivity * p.A_rad * (T^4 - T_space^4);

% Conduction from (warmer) structure to battery
Q_cond = p.G_structure * (T_structure - T);

% Joule heating (I²R for charge)
Q_joule = p.I_ch^2 * IR;

% Heater is less needed during sunlight; kept as backup
if T < p.T_heater_on
    P_heater = p.P_heater;
else
    P_heater = 0.0;
end

% Temperature derivative
dT = (Q_joule + Q_cond + P_heater - Q_rad) / (p.m * p.Cp);

% --- dSOH/dt: aging during charge (also ages battery) ---
dSOH = -p.alpha * p.I_ch * exp(-p.Ea / (p.R_gaz * T)) * stress * p.gamma;

dxdt = [dSOC; dT; dSOH];

% --- Terminal voltage (charge: IR adds to OCV) ---
V_term = p.OCV(SOC) + IR * p.I_ch;
V_term = max(p.V_pack_min, min(p.V_pack_max, V_term));

end
