function p = cubesat_params()
% =========================================================================

% --- Battery Pack (not single cell) ---
p.Q_nom = 7.5;            % Ah nominal (30 Wh / 4V midpoint ~ 7.5 Ah)
p.V_pack_min = 6.2;       % V (pack level, )
p.V_pack_max = 8.4;       % V (pack level, )
p.V_pack_mid = 7.4;       % V (nominal midpoint for OCV reference)

% --- Electrical behavior ---
p.I_dch = 8.0;            % A (discharge, realistic 3U EO mission eclipse load)
p.I_ch = 4.5;             % A (charge, solar-limited to match 55-min sunlight window)
p.eta = 0.98;             % Coulombic efficiency (pack level, includes balancing losses)

% --- OCV curve (Li-ion pack-representative, 3S)
% Mapping SOC [0,1] to V_pack [6.2, 8.4]
p.OCV = @(soc) p.V_pack_min + (p.V_pack_max - p.V_pack_min) .* ...
                (0.02 + 0.98*soc - 0.50*soc.^2 + 0.50*soc.^3);

% --- Internal Resistance (pack-level, includes all cells + wiring)
p.R0 = 0.15;              % Ω (pack-level initial resistance)
p.Ea = 32000;             % J/mol (activation energy, slightly higher for pack)
p.R_gaz = 8.314;          % Gas constant
p.T_ref = 298.15;         % K (25°C reference)

% --- Aging Parameters ---
p.alpha = 1.5e-3;         % Base aging rate (cycle/calendar degradation)
p.gamma = 1.0;            % Fault multiplier (modified by inject_fault)

% --- CORRECTED THERMAL MODEL 
% In vacuum, heat transfer = RADIATION + CONDUCTION, NOT convection
% Battery is thermally coupled to structure with insulation/radiator
p.m = 0.350;              % kg (pack + mounting, ~350 g)
p.Cp = 1050;              % J/kg·K (effective specific heat, Li-ion + structure)

% Radiation parameters (Stefan-Boltzmann)
p.emissivity = 0.85;      % Typical spacecraft coating
p.sigma = 5.67e-8;        % Stefan-Boltzmann constant (W/m²·K⁴)
p.A_rad = 0.015;          % m² (radiating surface area)

% Conduction to structure (thermal conductance)
p.G_structure = 2.0;      % W/K (conduction coupling to spacecraft structure)

% Heater control (thermostat-like behavior)
p.T_heater_on = 263.15;   % K (-10°C, turn on heater below this)
p.T_heater_off = 293.15;  % K (+20°C, turn off heater above this)
p.P_heater = 5.0;         % W (heater power when active, realistic for 3U)

% Environmental temperatures (LEO realistic, not extreme ladder)
% -10°C to +50°C represents managed pack temperature range
% Ambient will vary but heater + thermal control keep pack in band
p.T_operating_min = 263.15;  % K (-10°C, minimum operating)
p.T_operating_max = 323.15;  % K (+50°C, maximum operating)
p.T_amb_sun = 323.15;        % K (+50°C, full sunlight panel)
p.T_amb_eclipse = 263.15;    % K (-10°C, eclipse side)

% --- Orbit & Duty Cycle ---
p.orbit_period = 93 * 60;     % seconds (~93 min LEO)
p.eclipse_time = 35 * 60;     % seconds (~35 min eclipse)
p.sunlight_time = 58 * 60;    % seconds (~58 min sunlight)

% --- Limits ---
p.SOH_min = 0.70;             % Allow aging to 70% (conservative for mission)
p.SOC_min = 0.5;              % DoD = 50% (typical CubeSat practice)
p.SOC_max = 0.99;             % Charge to 99% SOC

% --- Mission assumptions documentation ---
p.mission_label = '3U Earth-observation CubeSat, sun-synchronous LEO, 525 km';
p.battery_class = 'AAC Clyde Space OPTIMUS-class, 30 Wh, Li-ion/LiPo, 2S-3S pack';
p.units_IR = 'Ohms';
p.units_Q = 'Ah';
p.units_V = 'V';
p.units_T = 'K';

end
