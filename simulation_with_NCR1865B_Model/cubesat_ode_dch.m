function [dxdt, V_term, IR] = cubesat_ode_dch(t, x, p)
% =========================================================================


% --- State bounds ---
SOC = max(p.SOC_min,          min(p.SOC_max,          x(1)));
T   = max(p.T_operating_min,  min(p.T_operating_max,  x(2)));
SOH = max(p.SOH_min,          min(1.05,               x(3)));

% --- Internal resistance (unchanged) ---
IR = (p.R0 / SOH) * (1 + p.k_R * (1 - SOH)) * ...
     exp((p.Ea / p.R_gaz) * (1/T - 1/p.T_ref));

% --- SOC stress factor (unchanged) ---
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% --- dSOC/dt (unchanged) ---
dSOC = -(p.eta * p.I_dch) / (p.Q_nom * SOH * 3600);

% --- dT/dt : vacuum thermal balance (unchanged) ---
T_space = 3.0;
Q_rad   = p.sigma * p.emissivity * p.A_rad * (T^4 - T_space^4);
Q_cond  = p.G_structure * (T - p.T_amb_eclipse);
Q_joule = p.I_dch^2 * IR;
P_heat  = p.P_heater * double(T < p.T_heater_on);
dT      = (Q_joule + P_heat - Q_rad - Q_cond) / (p.m * p.Cp);

% --- dSOH/dt — REALISTIC NON-LINEAR DEGRADATION ---
base_rate = p.alpha * p.I_dch * exp(-p.Ea / (p.R_gaz * T)) * stress * p.gamma;
sei_term  = p.sei_factor * exp(-p.sei_tau * (1 - SOH));   % fast initial SEI
accel_term= p.k_accel * (1 - SOH);                         % EOL acceleration
dSOH = -base_rate * (sei_term + 1.0 + accel_term);

dxdt = [dSOC; dT; dSOH];

% --- Terminal voltage (unchanged) ---
V_term = p.OCV(SOC) - IR * p.I_dch;
V_term = max(p.V_pack_min, min(p.V_pack_max, V_term));

end