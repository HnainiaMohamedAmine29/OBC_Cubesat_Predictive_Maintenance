function [dxdt, V_term, IR] = cubesat_ode_ch(t, x, p)
% ODE for CHARGE phase
% x = [SOC; T(K); SOH]
% I_ch is positive → SOC increases

% State limits
SOC = max(p.SOC_min, min(p.SOC_max, x(1)));
T   = max(123.0, min(393.0, x(2)));
SOH = max(p.SOH_min, min(1.05, x(3)));

% Internal resistance
IR = (p.R0 / SOH) * exp((p.Ea/p.R_gaz) * (1/T - 1/p.T_ref));

% Stress factor
stress = 0.10 + 0.50*exp(-14*SOC) + 0.40*exp(-14*(1-SOC));

% dSOC/dt
dSOC = (p.eta * p.I_ch) / (p.Q_nom * SOH * 3600);

% dT/dt
T_C = T - 273.15;
dT = (p.I_ch^2 * IR + p.h_conv * p.A_surf * (p.T_amb - T_C)) / (p.m * p.Cp);

% dSOH/dt (charging also ages battery)
dSOH = -p.alpha * p.I_ch * exp(-p.Ea/(p.R_gaz*T)) * stress * p.gamma;

dxdt = [dSOC; dT; dSOH];

% Terminal voltage (charging: IR drop adds)
V_term = p.OCV(SOC) + IR * p.I_ch;
V_term = max(p.V_min, min(p.V_max + 0.1, V_term));
end