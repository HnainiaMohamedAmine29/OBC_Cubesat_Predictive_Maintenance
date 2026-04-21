function p = cubesat_params()

% -------------------------------------------------------------------------
% Battery Pack 
% -------------------------------------------------------------------------
p.Q_nom      = 3.350;        % Ah   pack rated capacity at BOL
p.V_pack_min = 6.20;        % V
p.V_pack_max = 8.40;        % V
p.V_pack_mid = 7.20;        % V

% -------------------------------------------------------------------------
% Operating currents
% -------------------------------------------------------------------------
p.I_dch = 1.91;             % A    discharge current (eclipse load)
p.I_ch  = 1.2155;           % A    charge current (sunlight)
p.eta   = 0.98;             % -    Coulombic efficiency

% OCV polynomial (unchanged)
p.OCV = @(soc) p.V_pack_min + (p.V_pack_max - p.V_pack_min) .* ...
               (0.018 + 0.940.*soc - 0.380.*soc.^2 + 0.420.*soc.^3);

% -------------------------------------------------------------------------
% Internal Resistance  [1][2][5]
% -------------------------------------------------------------------------
p.R0     = 0.045;           % Ohm  BOL pack IR
p.k_R    = 1.90;            % -    resistance growth with degradation
p.Ea     = 31700;           % J/mol
p.R_gaz  = 8.314;           % J/mol.K
p.T_ref  = 298.15;          % K

% -------------------------------------------------------------------------
% Aging  
% -------------------------------------------------------------------------
p.alpha      = 0.15;        % base aging rate (tuned with SEI/accel terms)
p.gamma      = 1.0;         % fault multiplier
p.sei_factor = 2.0;         % SEI formation boost (fast initial aging)
p.sei_tau    = 12.0;        % SEI decay constant (higher = quicker transition to mid-life)
p.k_accel    = 5.0;         % EOL acceleration (sharp drop near 70% SOH)



% -------------------------------------------------------------------------
% Thermal model — vacuum (unchanged)
% -------------------------------------------------------------------------
p.m          = 0.200;       % kg
p.Cp         = 830;         % J/kg.K
p.sigma      = 5.67e-8;     % W/m2.K4
p.emissivity = 0.85;        % -
p.A_rad      = 0.004;       % m2
p.G_structure= 0.50;        % W/K
p.T_heater_on  = 263.15;    % K
p.T_heater_off = 293.15;    % K
p.P_heater     = 2.0;       % W

% -------------------------------------------------------------------------
% Environmental limits & orbit (unchanged)
% -------------------------------------------------------------------------
p.T_operating_min = 263.15;
p.T_operating_max = 323.15;
p.T_amb_sun        = 308.15;
p.T_amb_eclipse    = 263.15;

p.orbit_period  = 90 * 60;
p.eclipse_time  = 35 * 60;
p.sunlight_time = 55 * 60;

% -------------------------------------------------------------------------
% State limits
% -------------------------------------------------------------------------
p.SOH_min = 0.70;
p.SOH_eol = 0.70;
p.SOC_min = 0.50;
p.SOC_max = 0.99;

% -------------------------------------------------------------------------
% Labels (unchanged)
% -------------------------------------------------------------------------
p.cell_model    = 'Panasonic NCR18650B';
p.pack_config   = '2S1P';
p.chemistry     = 'NCA';
p.mission_label = 'LEO Earth-observation satellite, sun-synchronous orbit';
p.units_IR = 'Ohms';
p.units_Q  = 'Ah';
p.units_V  = 'V';
p.units_T  = 'K (stored) / C (exported)';

end