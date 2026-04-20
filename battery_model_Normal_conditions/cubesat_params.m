function p = cubesat_params()


% --- Battery Pack ---
p.Q_nom      = 4.035;       % Ah   pack rated capacity at BOL
p.V_pack_min = 6.20;        % V    pack discharged lower limit
p.V_pack_max = 8.40;        % V    pack charged upper limit
p.V_pack_mid = 7.40;        % V    nominal midpoint

% --- Operating currents ---

p.I_dch = 1.91;             % A   discharge current (eclipse)
p.I_ch  = 1.2155;           % A   charge current (sunlight)
%   I_ch = I_dch * t_eclipse / t_sunlight = 1.91 * 35/55 = 1.2155 A
%   This makes chargetime ≈ 55 min (SOH cancels in the ratio)
p.eta   = 0.98;             % -   Coulombic efficiency

p.OCV = @(soc) p.V_pack_min + (p.V_pack_max - p.V_pack_min) .* ...
               (0.02 + 0.98.*soc - 0.50.*soc.^2 + 0.50.*soc.^3);

% --- Internal Resistance (Arrhenius + capacity-fade growth) ---

%

p.R0     = 0.015;         % Ω   BOL pack resistance (55.20 mΩ)
p.k_R    = 1.899;           % -   resistance growth factor with degradation
p.Ea     = 32000;           % J/mol  activation energy
p.R_gaz  = 8.314;           % J/mol·K
p.T_ref  = 298.15;          % K   (25 °C reference)

% --- Aging ---


p.alpha = 0.02085;          % base aging rate
p.gamma = 1.0;              % fault multiplier (overwritten by inject_fault.m)

% --- Thermal model (VACUUM: conduction + radiation, NO convection) ---
p.m          = 0.350;       % kg   battery + mounting bracket
p.Cp         = 1050;        % J/kg·K
p.sigma      = 5.67e-8;     % W/m²·K⁴  Stefan–Boltzmann
p.emissivity = 0.85;
p.A_rad      = 0.004;       % m²  effective radiating area (interior mount)
p.G_structure= 0.50;        % W/K thermal conductance to structure
p.T_heater_on  = 263.15;    % K (-10 °C)
p.T_heater_off = 293.15;    % K (+20 °C)
p.P_heater     = 2.0;       % W

% --- Environmental ---
p.T_operating_min = 263.15; % K (-10 °C)
p.T_operating_max = 323.15; % K (+50 °C)
p.T_amb_sun        = 308.15;% K (+35 °C) structure in sunlight
p.T_amb_eclipse    = 263.15;% K (-10 °C) structure in eclipse

% --- Orbit (nominal; per-cycle values drawn in cubesat_run.m) ---
p.orbit_period  = 90 * 60;  % s
p.eclipse_time  = 35 * 60;  % s  nominal (35 min)
p.sunlight_time = 55 * 60;  % s  nominal (55 min)

% --- State limits ---
p.SOH_min = 0.70;           % 70% retained capacity = ODE lower clamp
p.SOH_eol = 0.70;           % End-of-Life threshold: simulation stops when
                             % SOH drops below this value, even if cycles remain.
                             % Standard Li-ion EOL = 70% (IEC 62660-1).
                             % Below this: voltage sag too large, mission power
                             % budget cannot be met, thermal runaway risk rises.
p.SOC_min = 0.50;           % absolute safety floor (normal DoD ~27%, floor well below)
p.SOC_max = 0.99;           % charge target

% --- Labels (generic — no platform names in exported data) ---
p.mission_label = 'LEO Earth-observation satellite, sun-synchronous orbit';
p.units_IR = 'Ohms';
p.units_Q  = 'Ah';
p.units_V  = 'V';
p.units_T  = 'K (stored) / C (exported)';

end
