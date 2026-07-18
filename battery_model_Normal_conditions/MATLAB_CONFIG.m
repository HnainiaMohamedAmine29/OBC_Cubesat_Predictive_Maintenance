# MATLAB Configuration for Battery Simulation
% ============================================================================
% OBC CubeSat Battery Simulation - Configurable Parameters
% Usage: Edit these values to customize your MATLAB simulation
% ============================================================================

% ============================================================================
% SIMULATION CONTROL
% ============================================================================

% Maximum number of orbital cycles to simulate
% Note: Simulation stops early if SOH drops below SOH_eol
N_MAX = 7000;

% Number of independent battery simulations (for Monte Carlo analysis)
N_BATT = 1;

% ============================================================================
% BATTERY PACK PARAMETERS
% ============================================================================

% Nominal (BOL) capacity
Q_NOM = 4.035;  % Ah

% Voltage limits
V_MIN = 6.20;   % V (discharged)
V_MAX = 8.40;   % V (charged)
V_MID = 7.40;   % V (nominal midpoint)

% ============================================================================
% OPERATING CURRENTS
% ============================================================================

% Discharge current during eclipse phase
I_DCH = 1.91;   % A

% Charge current during sunlight phase
I_CH = 1.2155;  % A
% Note: I_CH = I_DCH * (eclipse_time / sunlight_time) = 1.91 * 35/55

% Coulombic efficiency
ETA = 0.98;     % dimensionless

% ============================================================================
% INTERNAL RESISTANCE (Arrhenius Model)
% ============================================================================

% Baseline resistance at BOL (25°C)
R0 = 0.015;     % Ω (15 mΩ)

% Resistance growth factor with capacity fade
K_R = 1.899;    % dimensionless

% Activation energy
EA = 32000;     % J/mol

% Gas constant
R_GAS = 8.314;  % J/(mol·K)

% Reference temperature
T_REF = 298.15; % K (25°C)

% ============================================================================
% AGING MODEL (Capacity Fade)
% ============================================================================

% Base aging rate coefficient
ALPHA = 0.02085;    % dimensionless

% Fault severity multiplier (1.0 = normal, >1.0 = accelerated degradation)
GAMMA = 1.0;        % dimensionless

% End-of-life threshold (simulation stops when SOH drops below this)
SOH_EOL = 0.70;     % 70% (IEC 62660-1 standard)

% ============================================================================
% THERMAL MODEL (Vacuum Environment)
% ============================================================================

% Battery mass + mounting bracket
MASS = 0.350;   % kg

% Specific heat capacity
CP = 1050;      % J/(kg·K)

% Stefan-Boltzmann constant
SIGMA = 5.67e-8;    % W/(m²·K⁴)

% Emissivity
EMISSIVITY = 0.85;  % dimensionless

% Effective radiating area
A_RAD = 0.004;  % m² (interior mount)

% Thermal conductance to spacecraft structure
G_STRUCTURE = 0.50; % W/K

% ============================================================================
% HEATER CONTROL
% ============================================================================

% Temperature thresholds for heater
T_HEATER_ON = 263.15;   % K (-10°C) - turn on
T_HEATER_OFF = 293.15;  % K (+20°C) - turn off

% Heater power
P_HEATER = 2.0; % W

% ============================================================================
% ORBITAL ENVIRONMENT
% ============================================================================

% Orbital period
ORBIT_PERIOD = 90 * 60; % seconds (90 minutes)

% Eclipse duration (nominal, will vary per cycle)
ECLIPSE_TIME = 35 * 60; % seconds (~35 minutes)

% Sunlight duration (nominal, will vary per cycle)
SUNLIGHT_TIME = 55 * 60;    % seconds (~55 minutes)

% Structure temperature in sunlight
T_AMB_SUN = 308.15; % K (+35°C)

% Structure temperature in eclipse (nominal)
T_AMB_ECLIPSE = 263.15; % K (-10°C)

% Cold space temperature (for radiation model)
T_SPACE = 3;    % K

% ============================================================================
% OPERATING CONSTRAINTS
% ============================================================================

% Minimum operating temperature
T_OPERATING_MIN = 263.15;   % K (-10°C)

% Maximum operating temperature
T_OPERATING_MAX = 323.15;   % K (+50°C)

% State of Charge limits
SOC_MIN = 0.50; % 50% (safety floor)
SOC_MAX = 0.99; % 99% (charge target)

% ============================================================================
% OUTPUT & LOGGING
% ============================================================================

% Display progress every N cycles
PROGRESS_INTERVAL = 200;    % cycles

% Save output CSV file
SAVE_CSV = true;            % boolean

% CSV filename prefix
CSV_PREFIX = 'battery_dataset_normal';

% Enable validation checks
VALIDATE = true;            % boolean

% ============================================================================
% RANDOM SEED (for reproducibility)
% ============================================================================

% Set this to a fixed value for reproducible results, or [] for random
RANDOM_SEED = 42;

% ============================================================================
% END OF CONFIGURATION
% ============================================================================
