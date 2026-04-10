%% cubesat_run.m - CubeSat battery dataset with extreme temperatures
% 3000 cycles, one battery, mixed faults, voltage features included.
clear; close all; clc

p = cubesat_params();
p.T_amb_cycles = [-150, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120];  % °C

N_max = 3000;
battery_id = 'CUBESAT_001';


fault_types = { ...
    'normal',           70; ...
    'high_temperature', 15; ...
    'thermal_runaway',   5; ...
    'low_temperature',   5; ...
    'capacity_fade',     5};

% Preallocation (numeric arrays)
n = N_max;
battID = repmat({battery_id}, n, 1);
cycleNums = (1:n)';

IRs = zeros(n,1);
QCs = zeros(n,1);
QDs = zeros(n,1);
V_means = zeros(n,1);
V_mins = zeros(n,1);
V_maxs = zeros(n,1);
Tavgs = zeros(n,1);
Tmins = zeros(n,1);
Tmaxs = zeros(n,1);
chargetimes = zeros(n,1);
discharge_times = zeros(n,1);
sohs = zeros(n,1);
T_ambs = zeros(n,1);

% Initial state: [SOC; T(K); SOH]
x = [1.0; p.T_amb_cycles(1)+273.15; 1.0];

fprintf('=====================================================\n');
fprintf('  CubeSat Battery Simulation - %d cycles\n', N_max);
fprintf('  Temperature: %d°C to %d°C\n', min(p.T_amb_cycles), max(p.T_amb_cycles));
fprintf('=====================================================\n\n');

for cycle = 1:n
    % Select fault type for this cycle
    r = rand() * 100;
    cumsum = 0;
    selected_fault = 'normal';
    for i = 1:size(fault_types,1)
        cumsum = cumsum + fault_types{i,2};
        if r <= cumsum
            selected_fault = fault_types{i,1};
            break;
        end
    end
    FAULT{cycle} = selected_fault;

    % Ambient temperature from LEO cycle
    T_idx = mod(cycle-1, numel(p.T_amb_cycles)) + 1;
    T_amb = p.T_amb_cycles(T_idx);
    T_ambs(cycle) = T_amb;

    % Inject fault (modifies p parameters except T_amb)
    p_cycle = inject_fault(p, selected_fault);
    p_cycle.T_amb = T_amb;

    % Accelerate degradation in last 200 cycles (>=2800)
    if cycle >= 2800
        p_cycle.alpha = 4.0e-3;   % ↑ aging rate
        p_cycle.gamma = 8.0;      % ↑ fault multiplier
    end

    % Simulate one cycle (pass current cycle number for dynamic times)
    row = simulate_cycle(x, p_cycle, selected_fault, cycle);

    % Store results
    IRs(cycle) = row.IR;
    QCs(cycle) = row.QC;
    QDs(cycle) = row.QD;
    V_means(cycle) = row.V_mean;
    V_mins(cycle) = row.V_min;
    V_maxs(cycle) = row.V_max;
    Tavgs(cycle) = row.Tavg;
    Tmins(cycle) = row.Tmin;
    Tmaxs(cycle) = row.Tmax;
    chargetimes(cycle) = row.chargetime;
    discharge_times(cycle) = row.discharge_time;
    sohs(cycle) = row.SOH_end;

    % Propagate state to next cycle
    x = row.x_next;

    % Progress report
    if mod(cycle, 100) == 0 || cycle == 1 || cycle == n
        fprintf('Cycle %4d/%d: %-18s | SOH=%.4f | T_amb=%4d°C | V=%.3f V\n', ...
            cycle, n, selected_fault, sohs(cycle), T_amb, V_means(cycle));
    end
end

% Compute cycle life (first cycle where SOH < 0.80)
idx_eol = find(sohs < 0.80, 1);
% if isempty(idx_eol)
%     cl_val = n;
% else
%     cl_val = idx_eol;
% end
% cycle_life = repmat(cl_val, n, 1);

% Build table with 14 features (+ fault_type)
T = table(battID, cycleNums, IRs, QCs, QDs, V_means, V_mins, V_maxs, ...
    Tavgs, Tmins, Tmaxs, chargetimes, discharge_times, sohs, ...
    'VariableNames', {'battery_id','cycle','IR','QC','QD','V_mean','V_min','V_max', ...
                      'Tavg','Tmin','Tmax','chargetime','discharge_time','soh'});


% Export CSV
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
csv_filename = sprintf('battery_cycle_level_dataset_cubesat_%s.csv', timestamp);
writetable(T, csv_filename);

fprintf('\n✓ Dataset exported: %d cycles | 1 battery\n', height(T));
fprintf('  File: %s\n', csv_filename);
fprintf('  SOH: %.4f → %.4f | Cycle_Life: %d cycles\n', sohs(1), sohs(end), cl_val);
fprintf('  T: mean=%.1f°C | min=%.1f°C | max=%.1f°C\n', mean(T_ambs), min(T_ambs), max(T_ambs));
fprintf('  V: mean=%.3f V | min=%.3f V | max=%.3f V\n', mean(V_means), min(V_mins), max(V_maxs));
fprintf('=====================================================\n');

% Show first rows
fprintf('\nFirst 10 rows of dataset:\n');
disp(T(1:min(10,end), :));

% Fault distribution
groups = {'normal','high_temperature','thermal_runaway','low_temperature','capacity_fade'};
fprintf('\nFault distribution:\n');
for i = 1:numel(groups)
    cnt = sum(strcmp(FAULT, groups{i}));
    pct = 100 * cnt / n;
    fprintf('  %-20s: %4d cycles (%5.1f%%)\n', groups{i}, cnt, pct);
end