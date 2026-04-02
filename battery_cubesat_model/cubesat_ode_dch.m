function [dxdt, V_term, IR] = cubesat_ode_dch(t, x, p)
% ODE for DISCHARGE phase
% x = [SOC; T(K); SOH]
% I_dch is positive → SOC decreases

% State limits (support -150°C to +120°C)
SOC = max(p.SOC_min, min(p.SOC_max, x(1)));
T   = max(123.0, min(393.0, x(2)));    % -150°C to 120°C in Kelvin
SOH = max(p.SOH_min, min(1.05, x(3)));

% Internal resistance (Arrhenius)
IR = (p.R0 / SOH) * exp((p.Ea/p.R_gaz) * (1/T - 1/p.T_ref));

% SOC stress factor (U-shaped)
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% dSOC/dt
dSOC = -(p.eta * p.I_dch) / (p.Q_nom * SOH * 3600);

% dT/dt: Joule heating + convection
T_C = T - 273.15;
dT = (p.I_dch^2 * IR + p.h_conv * p.A_surf * (p.T_amb - T_C)) / (p.m * p.Cp);

% dSOH/dt: aging driven by current, temperature, stress, and fault gamma
% Higher temperature → exponentially higher aging rate
dSOH = -p.alpha * p.I_dch * exp(-p.Ea/(p.R_gaz*T)) * stress * p.gamma;

dxdt = [dSOC; dT; dSOH];

% Terminal voltage
V_term = p.OCV(SOC) - IR * p.I_dch;
V_term = max(p.V_min, min(p.V_max, V_term));
end