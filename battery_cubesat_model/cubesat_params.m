function p = cubesat_params()
% CubeSat Li-ion battery - realistic LEO extreme temperatures
% Orbit: 93 minutes → 36 min eclipse (discharge), 57 min sun (charge)
% Temperature range: -150°C (eclipse) to +120°C (sun)

% --- Electrical ---
p.Q_nom = 1.862;      % Ah (NASA B0005)
p.I_dch = 1.5;        % A (discharge)
p.I_ch  = 0.96;       % A (charge) → ~57 min for 0.5→0.99 SOC
p.eta   = 0.9998;
p.V_min = 2.7;
p.V_max = 4.2;

% OCV curve (18650)
p.OCV = @(soc) 2.80 + 2.10.*soc - 1.60.*soc.^2 + 0.95.*soc.^3;

% Internal resistance (NASA B0005)
p.R0    = 0.8;      % Ω (initial)
p.Ea    = 29000;      % J/mol (activation energy)
p.R_gaz = 8.314;
p.T_ref = 298.15;     % K (25°C reference)

% --- Thermal ---
p.m      = 0.047;     % kg
p.Cp     = 830;       % J/kg·K
p.h_conv = 2.0;       % W/m²·K (space vacuum convection low)
p.A_surf = 0.0045;    % m²

% --- Aging ---
p.alpha  = 2.0e-3;    % base aging coefficient
p.gamma  = 1.0;       % fault multiplier (adjusted by inject_fault)

% --- Environmental ---
p.Qsol = 1361;     % W/m² (solar constant at 1 AU, space)

% --- Limits ---
p.SOH_min = 0.05;     % allow deep aging for 3000 cycles
p.SOC_min = 0.5;      % DOD 50% (CubeSat typical)
p.SOC_max = 0.99;

% --- LEO extreme temperature cycle ---
% Orbit: -150°C (eclipse) → +120°C (sun hot spot)
% These are ambient; actual battery T tracks with delay
p.T_amb_cycles = [-150, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120]; % °C
end