%% CubeSat Battery Dataset Generator (CORRECTED)

clear; close all; clc

% Load mission parameters
p = cubesat_params();

% Simulation configuration
N_max = 3000;
n_batteries = 1;  % Can increase for multiple battery Monte Carlo

battery_ids = {};
for b = 1:n_batteries
    battery_ids{b} = sprintf('CUBESAT_EO_%03d', b);
end

% Fault scenario configuration (persistent, not random each cycle)
fault_scenarios = { ...
    'normal',           60; ...          % 60% normal operation
    'high_temperature', 15; ...          % 15% with cooling issues
    'low_temperature',   5; ...          % 5% with heater failures
    'capacity_fade',    15; ...          % 15% with material degradation
    'thermal_runaway',   5};             % 5% catastrophic (terminal)

fprintf('=====================================================\n');
fprintf('CubeSat Battery Simulation - CORRECTED VERSION\n');
fprintf('=====================================================\n');
fprintf('Mission: 3U EO, sun-synchronous LEO, 525 km\n');
fprintf('Battery: 30 Wh pack, 7.5 Ah nominal\n');
fprintf('Cycles: %d per battery\n', N_max);
fprintf('Batteries: %d\n', n_batteries);
fprintf('Thermal: Radiation/conduction (vacuum model)\n');
fprintf('=====================================================\n\n');

% =========================================================================
% MAIN SIMULATION LOOP
% =========================================================================

for batt_idx = 1:n_batteries
    battery_id = battery_ids{batt_idx};
    
    % Preallocate storage
    n = N_max;
    battID = repmat({battery_id}, n, 1);
    cycleNums = (1:n)';
    
    % Feature columns
    SOC_starts = zeros(n, 1);
    SOC_ends = zeros(n, 1);
    DoDs = zeros(n, 1);
    IRs = zeros(n, 1);
    QCs = zeros(n, 1);
    QDs = zeros(n, 1);
    V_means = zeros(n, 1);
    V_mins = zeros(n, 1);
    V_maxs = zeros(n, 1);
    Tavgs = zeros(n, 1);
    Tmins = zeros(n, 1);
    Tmaxs = zeros(n, 1);
    chargetimes = zeros(n, 1);
    discharge_times = zeros(n, 1);
    sohs = zeros(n, 1);
    T_ambs = zeros(n, 1);
    FAULT = cell(n, 1);
    fault_severities = zeros(n, 1);
    
    % Initial state: [SOC=1.0; T=start temp; SOH=1.0]
    T_init = p.T_operating_min + 5;  % Start at -5°C (realistic eclipse temp)
    x = [1.0; T_init; 1.0];
    
    % Persistent fault state (carries over cycles)
    fault_state.type = 'normal';
    fault_state.severity = 0.0;
    fault_state.cycle_start = 1;
    fault_state.is_terminal = false;
    
    fprintf('Battery %s:\n', battery_id);
    
    for cycle = 1:n
        
        % --- SELECT FAULT TYPE (persistent: may stay same as previous) ---
        if cycle == 1
            % First cycle: select initial fault
            r = rand() * 100;
            cumsum = 0;
            selected_fault = 'normal';
            for i = 1:size(fault_scenarios, 1)
                cumsum = cumsum + fault_scenarios{i, 2};
                if r <= cumsum
                    selected_fault = fault_scenarios{i, 1};
                    break;
                end
            end
            fault_state.type = selected_fault;
            fault_state.cycle_start = 1;
        else
            % Probability of fault transition (low, to maintain persistence)
            fault_transition_prob = 0.005;  % 0.5% chance to transition faults
            if rand() < fault_transition_prob && ~fault_state.is_terminal
                % Switch to a new fault
                r = rand() * 100;
                cumsum = 0;
                for i = 1:size(fault_scenarios, 1)
                    cumsum = cumsum + fault_scenarios{i, 2};
                    if r <= cumsum
                        selected_fault = fault_scenarios{i, 1};
                        break;
                    end
                end
                fault_state.type = selected_fault;
                fault_state.severity = 0.0;  % Reset severity for new fault
                fault_state.cycle_start = cycle;
            else
                % Continue current fault
                selected_fault = fault_state.type;
            end
        end
        
        % Inject fault (modifies parameters, updates fault_state)
        [p_cycle, fault_state] = inject_fault(p, selected_fault, fault_state, cycle);
        
        % Ambient temperature cycles (realistic LEO with heater control)
        % Use stored T_amb from params (which includes heater logic)
        if mod(cycle, 2) == 1
            p_cycle.T_amb_eclipse = p.T_operating_min;  % Eclipse: cold
        else
            p_cycle.T_amb_eclipse = p.T_operating_min + 10;  % Between orbits: warm
        end
        
        % Accelerate degradation in last 200 cycles (optional, realistic aging)
        if cycle >= 2800
            p_cycle.alpha = p_cycle.alpha * 2.0;   % 2x aging rate near EOL
            p_cycle.gamma = p_cycle.gamma * 1.5;   % 1.5x fault multiplier
        end
        
        % --- SIMULATE ONE CYCLE ---
        row = simulate_cycle(x, p_cycle, selected_fault, cycle);
        
        % --- STORE RESULTS ---
        SOC_starts(cycle) = row.SOC_start;
        SOC_ends(cycle) = row.SOC_end;
        DoDs(cycle) = row.DoD;
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
        T_ambs(cycle) = p_cycle.T_amb_eclipse;
        FAULT{cycle} = row.fault_type;
        fault_severities(cycle) = fault_state.severity;
        
        % Propagate state to next cycle
        x = row.x_next;
        
        % --- PROGRESS REPORT ---
        if mod(cycle, 200) == 0 || cycle == 1 || cycle == n
            fprintf('  Cycle %4d/%d: %-18s | SOH=%.4f | V=%.3f V | T=%5.1f°C | QC=%.3f Ah\n', ...
                cycle, n, FAULT{cycle}, sohs(cycle), V_means(cycle), Tavgs(cycle), QCs(cycle));
        end
        
        % Early termination if battery is terminal (thermal runaway reached end)
        if fault_state.is_terminal && sohs(cycle) < 0.50
            fprintf('  ⚠ Thermal runaway detected at cycle %d. Terminating simulation.\n', cycle);
            % Trim arrays to actual cycles completed
            n = cycle;
            SOC_starts = SOC_starts(1:n);
            SOC_ends = SOC_ends(1:n);
            DoDs = DoDs(1:n);
            IRs = IRs(1:n);
            QCs = QCs(1:n);
            QDs = QDs(1:n);
            V_means = V_means(1:n);
            V_mins = V_mins(1:n);
            V_maxs = V_maxs(1:n);
            Tavgs = Tavgs(1:n);
            Tmins = Tmins(1:n);
            Tmaxs = Tmaxs(1:n);
            chargetimes = chargetimes(1:n);
            discharge_times = discharge_times(1:n);
            sohs = sohs(1:n);
            T_ambs = T_ambs(1:n);
            FAULT = FAULT(1:n);
            fault_severities = fault_severities(1:n);
            cycleNums = cycleNums(1:n);
            battID = battID(1:n);
            break;
        end
    end
    
    % --- CYCLE LIFE CALCULATION ---
    % Find first cycle where SOH drops below 80% (standard EoL definition)
    idx_eol_80 = find(sohs < 0.80, 1);
    if isempty(idx_eol_80)
        cycle_life_80 = n;  % Didn't reach 80% within simulation
    else
        cycle_life_80 = idx_eol_80;
    end
    
    % Also report 70% EOL
    idx_eol_70 = find(sohs < 0.70, 1);
    if isempty(idx_eol_70)
        cycle_life_70 = n;
    else
        cycle_life_70 = idx_eol_70;
    end
    
    % --- VALIDATION CHECKS ---
    fprintf('\n  Validation checks:\n');
    n_finite = sum(isfinite(QCs)) + sum(isfinite(QDs)) + sum(isfinite(IRs));
    fprintf('    Finite values: %d/%d\n', n_finite, 3*n);
    
    n_qc_valid = sum(QCs > 0 & QCs < 10);
    fprintf('    Valid QC (0-10 Ah): %d/%d\n', n_qc_valid, n);
    
    n_temp_valid = sum(Tavgs >= -10 & Tavgs <= 50);
    fprintf('    Valid T (-10 to +50°C): %d/%d\n', n_temp_valid, n);
    
    n_volt_valid = sum(V_means >= p.V_pack_min & V_means <= p.V_pack_max);
    fprintf('    Valid V (%.1f-%.1f V): %d/%d\n', p.V_pack_min, p.V_pack_max, n_volt_valid, n);
    
    % --- BUILD EXPORT TABLE ---
    T = table(battID(1:n), cycleNums, SOC_starts, SOC_ends, DoDs, ...
        IRs, QCs, QDs, V_means, V_mins, V_maxs, Tavgs, Tmins, Tmaxs, ...
        chargetimes, discharge_times, sohs, T_ambs, FAULT, fault_severities, ...
        'VariableNames', {'battery_id', 'cycle', 'SOC_start', 'SOC_end', 'DoD', ...
                          'IR', 'QC', 'QD', 'V_mean', 'V_min', 'V_max', ...
                          'Tavg', 'Tmin', 'Tmax', 'chargetime_min', 'discharge_time_min', ...
                          'soh', 'T_amb', 'fault_type', 'fault_severity'});
    
    % --- EXPORT CSV ---
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    csv_filename = sprintf('battery_dataset_cubesat_corrected_%s.csv', timestamp);
    writetable(T, csv_filename);
    
    % --- SUMMARY REPORT ---
    fprintf('\n  Summary for %s:\n', battery_id);
    fprintf('    Total cycles simulated: %d\n', n);
    fprintf('    Cycle life (SOH 80%%): %d cycles\n', cycle_life_80);
    fprintf('    Cycle life (SOH 70%%): %d cycles\n', cycle_life_70);
    fprintf('    SOH: %.4f → %.4f\n', sohs(1), sohs(end));
    fprintf('    Temperature: mean=%.1f°C, range=[%.1f, %.1f]°C\n', ...
        mean(Tavgs), min(Tmins), max(Tmaxs));
    fprintf('    Voltage: mean=%.3f V, range=[%.3f, %.3f] V\n', ...
        mean(V_means), min(V_mins), max(V_maxs));
    fprintf('    Capacity: QC mean=%.3f Ah, QD mean=%.3f Ah\n', mean(QCs), mean(QDs));
    fprintf('    Resistance: IR mean=%.3f Ω (increasing from aging)\n', mean(IRs));
    
    % Fault distribution
    fprintf('\n  Fault distribution:\n');
    fault_types = unique(FAULT);
    for i = 1:length(fault_types)
        cnt = sum(strcmp(FAULT, fault_types{i}));
        pct = 100 * cnt / n;
        fprintf('    %-20s: %4d cycles (%5.1f%%)\n', fault_types{i}, cnt, pct);
    end
    
    % Export summary
    fprintf('\n  ✓ Dataset exported: %s\n', csv_filename);
    fprintf('    Features: %d columns, %d rows\n', width(T), height(T));
    fprintf('    Includes: SOC_start/end, DoD, IR, QC/QD, Voltages, Temps, Times,\n');
    fprintf('              SOH, T_amb, fault_type, fault_severity\n');
    
    % Display first rows
    fprintf('\n  First 10 rows:\n');
    disp(T(1:min(10, end), :));
    
end  % End battery loop

fprintf('\n=====================================================\n');
fprintf('Simulation complete.\n');
fprintf('=====================================================\n');
