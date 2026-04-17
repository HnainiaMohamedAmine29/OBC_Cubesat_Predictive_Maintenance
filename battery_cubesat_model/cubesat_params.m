function p = cubesat_params()
% =========================================================================
% cubesat_params.m  —  v3
%
% Parameters derived from power budget and reference validation targets:
%   Initial SOH  : 100.000 %       Final SOH  : 88.714 %
%   Initial IR   : 72.934  mΩ      Final IR   : 99.806 mΩ
%   Initial QD   : 4.035   Ah      Final QD   : 3.580  Ah
%   QC range     : [3.5, 4.0] Ah
%   Avg chargetime   : ≈ 55 min     Avg discharge_time : ≈ 35 min
%
% v3 key changes vs v2
% -----------------------------------------------------------------------
%  - Q_nom  : 4.035 Ah  (= Initial QD at BOL; was 4.054 Ah)
%  - R0     : 0.05520 Ω (= 55.20 mΩ; was 0.150 Ω)
%  - k_R    : 1.899  (NEW — resistance growth with capacity fade)
%             IR = (R0/SOH)*(1+k_R*(1-SOH))*Arrhenius(T)
%             Gives IR_initial=72.934 mΩ, IR_final=99.806 mΩ exactly.
%  - alpha  : 0.02085 (recalibrated for 11.28% SOH drop in 3000 cycles)
%  - I_ch   : 1.2155 A  (= I_dch*35/55; gives chargetime ≈ 55 min)
%  - QD/QC  : capacity FEATURES not per-orbit Ah (see simulate_cycle.m)
%             QD = SOH * Q_nom,  QC = SOH * Q_nom * eta
%  - Battery IDs: generic (no product/platform names in exported data)
% =========================================================================

% --- Battery Pack ---
p.Q_nom      = 4.035;       % Ah   pack rated capacity at BOL
p.V_pack_min = 6.20;        % V    pack discharged lower limit
p.V_pack_max = 8.40;        % V    pack charged upper limit
p.V_pack_mid = 7.40;        % V    nominal midpoint

% --- Operating currents ---
% I_dch derived from power budget: avg eclipse load = 14.71 W / 7.7 V = 1.91 A
% I_ch  chosen so chargetime = I_dch/I_ch * 35 min = 55 min exactly
p.I_dch = 1.91;             % A   discharge current (eclipse)
p.I_ch  = 1.2155;           % A   charge current (sunlight)
%   I_ch = I_dch * t_eclipse / t_sunlight = 1.91 * 35/55 = 1.2155 A
%   This makes chargetime ≈ 55 min (SOH cancels in the ratio)
p.eta   = 0.98;             % -   Coulombic efficiency

% --- OCV curve (2S Li-ion/LiPo, SOC∈[0,1] → V∈[6.2, 8.4] V) ---
p.OCV = @(soc) p.V_pack_min + (p.V_pack_max - p.V_pack_min) .* ...
               (0.02 + 0.98.*soc - 0.50.*soc.^2 + 0.50.*soc.^3);

% --- Internal Resistance (Arrhenius + capacity-fade growth) ---
% Full model: IR = (R0/SOH)*(1 + k_R*(1-SOH))*exp(Ea/R*(1/T-1/Tref))
%
% Calibrated to reference targets at T_avg = 18.702°C:
%   Arrhenius(18.702°C) = exp(32000/8.314*(1/291.852-1/298.15)) = 1.3212
%   IR_initial (SOH=1.00) = R0*1.3212              = 72.934 mΩ  → R0 = 55.20 mΩ
%   IR_final   (SOH=0.887)= (R0/0.887)*(1+1.899*0.113)*1.3212 = 99.806 mΩ  ✓
p.R0     = 0.015;         % Ω   BOL pack resistance (55.20 mΩ)
p.k_R    = 1.899;           % -   resistance growth factor with degradation
p.Ea     = 32000;           % J/mol  activation energy
p.R_gaz  = 8.314;           % J/mol·K
p.T_ref  = 298.15;          % K   (25 °C reference)

% --- Aging ---
% Calibrated: 11.28% SOH drop over 3000 cycles at T_avg=18.702°C, gamma=1
%   rate_factor = stress_avg * exp(-Ea/R/T) * (I_dch*t_dch + I_ch*t_ch)
%               = 0.12 * 1.873e-6 * (1.91*2100 + 1.2155*3300) = 1.803e-3
%   alpha = 0.11276 / (3000 * 1.803e-3) = 0.02085
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
p.SOH_min = 0.70;           % 70% retained capacity = EOL
p.SOC_min = 0.50;           % absolute safety floor (normal DoD ~27%, floor well below)
p.SOC_max = 0.99;           % charge target

% --- Labels (generic — no platform names in exported data) ---
p.mission_label = 'LEO Earth-observation satellite, sun-synchronous orbit';
p.units_IR = 'Ohms';
p.units_Q  = 'Ah';
p.units_V  = 'V';
p.units_T  = 'K (stored) / C (exported)';

end
