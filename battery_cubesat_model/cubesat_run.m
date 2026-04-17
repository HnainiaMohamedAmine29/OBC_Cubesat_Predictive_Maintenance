%% cubesat_run.m  —  v3
% =========================================================================

clear; close all; clc

p      = cubesat_params();
N_max  = 3000;
n_batt = 1;          % increase for multi-battery Monte-Carlo

battery_ids = {};
for b = 1:n_batt
    battery_ids{b} = sprintf('BATT_EO_%03d', b);
end

% Fault scenario weights (persistent; 0.5% per-cycle transition probability)
fault_scenarios = { ...
    'normal',           60; ...   % 60%  baseline
    'high_temperature', 15; ...   % 15%  cooling degradation
    'low_temperature',   5; ...   %  5%  heater failure
    'capacity_fade',    15; ...   % 15%  material degradation
    'thermal_runaway',   5};      %  5%  catastrophic (terminal)

fprintf('=====================================================\n');
fprintf('Battery Cycle Dataset Generator  —  v3\n');
fprintf('=====================================================\n');
fprintf('Q_nom   = %.3f Ah   R0 = %.2f mΩ   k_R = %.3f\n', ...
        p.Q_nom, p.R0*1000, p.k_R);
fprintf('alpha   = %.5f    I_dch = %.2f A   I_ch = %.4f A\n', ...
        p.alpha, p.I_dch, p.I_ch);
fprintf('QD=SOH*Q_nom   QC=SOH*Q_nom*eta\n');
fprintf('Eclipse ~ N(35,1)min   Sunlight = 55 min\n');
fprintf('Battery IDs: generic (BATT_EO_NNN)\n');
fprintf('=====================================================\n\n');

% =========================================================================
for batt_idx = 1:n_batt

    battery_id = battery_ids{batt_idx};
    rng(batt_idx * 42);

    n = N_max;

    % --- Preallocate ---
    battID           = repmat({battery_id}, n, 1);
    cycleNums        = (1:n)';
    SOC_starts       = zeros(n,1);
    SOC_ends         = zeros(n,1);
    DoDs             = zeros(n,1);
    IRs              = zeros(n,1);
    QDs              = zeros(n,1);
    QCs              = zeros(n,1);
    V_means          = zeros(n,1);
    V_mins           = zeros(n,1);
    V_maxs           = zeros(n,1);
    Tavgs            = zeros(n,1);
    Tmins            = zeros(n,1);
    Tmaxs            = zeros(n,1);
    chargetimes      = zeros(n,1);
    discharge_times  = zeros(n,1);
    sohs             = zeros(n,1);
    T_ambs           = zeros(n,1);
    FAULT            = cell(n,1);

    % Initial state: SOC=0.99, T=−5°C, SOH=1.0
    x = [0.99; p.T_operating_min + 5.0; 1.0];

    fault_state.type        = 'normal';
    fault_state.severity    = 0.0;
    fault_state.cycle_start = 1;
    fault_state.is_terminal = false;

    fprintf('Battery %s:\n', battery_id);

    for cycle = 1:n

        % =================================================================
        % FAULT SELECTION (persistent; 0.5% transition probability)
        % =================================================================
        if cycle == 1
            selected_fault   = sample_fault(fault_scenarios);
            fault_state.type = selected_fault;
        else
            if rand() < 0.005 && ~fault_state.is_terminal
                selected_fault          = sample_fault(fault_scenarios);
                fault_state.type        = selected_fault;
                fault_state.severity    = 0.0;
                fault_state.cycle_start = cycle;
            else
                selected_fault = fault_state.type;
            end
        end

        % =================================================================
        % FAULT INJECTION
        % =================================================================
        [p_cycle, fault_state] = inject_fault(p, selected_fault, fault_state, cycle);

        % =================================================================
        % PER-CYCLE ORBIT TIMES
        %
        % Eclipse ~ N(35, 1) min, clipped [32, 38] min
        %   → discharge_time varies; QD/QC unchanged (they use SOH, not t)
        % Sunlight = 55 min (fixed) so chargetime ≈ 55 min on average
        %
        % Fault modifiers (physical rationale):
        %   high_temperature → +2 min sunlight exposure (cooling degraded)
        %   low_temperature  → +2 min eclipse cold soak (heater failed)
        % =================================================================
        eclipse_mean  = 35.0;
        if strcmp(selected_fault, 'high_temperature')
            eclipse_mean = eclipse_mean - 1.0;  % slightly shorter eclipse
        elseif strcmp(selected_fault, 'low_temperature')
            eclipse_mean = eclipse_mean + 2.0;  % longer cold eclipse
        end
        eclipse_min  = max(32.0, min(38.0, eclipse_mean + 1.0 * randn()));
        sunlight_min = 55.0;   % fixed — gives chargetime ≈ 55 min avg

        p_cycle.eclipse_time  = eclipse_min  * 60;
        p_cycle.sunlight_time = sunlight_min * 60;

        % Alternate eclipse structure temperature (orbit-to-orbit variation)
        if mod(cycle, 2) == 1
            p_cycle.T_amb_eclipse = p.T_operating_min;         % −10 °C
        else
            p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;  %  −2 °C
        end

        % =================================================================
        % SIMULATE ONE ORBIT
        % =================================================================
        row = simulate_cycle(x, p_cycle, selected_fault, cycle);

        % =================================================================
        % STORE (QD and QC are capacity features from simulate_cycle)
        % =================================================================
        SOC_starts(cycle)      = row.SOC_start;
        SOC_ends(cycle)        = row.SOC_end;     % SOC after DISCHARGE
        DoDs(cycle)            = row.DoD;
        IRs(cycle)             = row.IR_ohm;
        QDs(cycle)             = row.QD_Ah;       % SOH * Q_nom
        QCs(cycle)             = row.QC_Ah;       % SOH * Q_nom * eta
        V_means(cycle)         = row.V_mean_V;
        V_mins(cycle)          = row.V_min_V;
        V_maxs(cycle)          = row.V_max_V;
        Tavgs(cycle)           = row.Tavg_C;
        Tmins(cycle)           = row.Tmin_C;
        Tmaxs(cycle)           = row.Tmax_C;
        chargetimes(cycle)     = row.chargetime_min;
        discharge_times(cycle) = row.discharge_time_min;
        sohs(cycle)            = row.SOH_end;
        T_ambs(cycle)          = p_cycle.T_amb_eclipse;
        FAULT{cycle}           = row.fault_type;

        % x_next: state at END of charge → SOC_start of next cycle
        x = row.x_next;

        % Progress print
        if mod(cycle, 200) == 0 || cycle == 1 || cycle == n
            fprintf(['  Cycle %4d/%d | %-18s | SOH=%.4f | ' ...
                     'SOC %.3f→%.3f (DoD=%.1f%%) | ' ...
                     'QD=%.3fAh QC=%.3fAh | ' ...
                     'IR=%.1fmΩ | t_d=%.1fm t_c=%.1fm\n'], ...
                cycle, n, FAULT{cycle}, sohs(cycle), ...
                SOC_starts(cycle), SOC_ends(cycle), DoDs(cycle)*100, ...
                QDs(cycle), QCs(cycle), IRs(cycle)*1000, ...
                discharge_times(cycle), chargetimes(cycle));
        end

        % Early exit on terminal failure
        if fault_state.is_terminal && sohs(cycle) < 0.50
            fprintf('  WARNING: Terminal failure at cycle %d.\n', cycle);
            n = cycle;
            SOC_starts      = SOC_starts(1:n);
            SOC_ends        = SOC_ends(1:n);
            DoDs            = DoDs(1:n);
            IRs             = IRs(1:n);
            QDs             = QDs(1:n);
            QCs             = QCs(1:n);
            V_means         = V_means(1:n);
            V_mins          = V_mins(1:n);
            V_maxs          = V_maxs(1:n);
            Tavgs           = Tavgs(1:n);
            Tmins           = Tmins(1:n);
            Tmaxs           = Tmaxs(1:n);
            chargetimes     = chargetimes(1:n);
            discharge_times = discharge_times(1:n);
            sohs            = sohs(1:n);
            T_ambs          = T_ambs(1:n);
            FAULT           = FAULT(1:n);
            cycleNums       = cycleNums(1:n);
            battID          = battID(1:n);
            break;
        end
    end

    % =====================================================================
    % CYCLE LIFE
    % =====================================================================
    idx80 = find(sohs < 0.80, 1);
    cycle_life_80 = n; if ~isempty(idx80), cycle_life_80 = idx80; end
    idx70 = find(sohs < 0.70, 1);
    cycle_life_70 = n; if ~isempty(idx70), cycle_life_70 = idx70; end

    % =====================================================================
    % VALIDATION GATES
    % =====================================================================
    fprintf('\n  Validation:\n');
    fprintf('    Cycles completed      : %d/%d\n', n, N_max);
    fprintf('    Finite (QD,QC,IR)     : %d/%d\n', ...
        sum(isfinite(QDs)&isfinite(QCs)&isfinite(IRs)), n);
    fprintf('    QD in [%.2f, %.2f] Ah : %d/%d\n', ...
        p.Q_nom*p.SOH_min, p.Q_nom, sum(QDs>=p.Q_nom*p.SOH_min & QDs<=p.Q_nom*1.01), n);
    fprintf('    QC in [3.50, 4.00] Ah : %d/%d\n', ...
        sum(QCs >= 3.50 & QCs <= 4.00), n);
    fprintf('    DoD = SOC_s-SOC_e     : %d/%d consistent\n', ...
        sum(abs(SOC_starts-SOC_ends-DoDs) < 1e-5), n);
    fprintf('    T  in [-10, +50] C    : %d/%d\n', ...
        sum(Tavgs >= -10 & Tavgs <= 50), n);
    fprintf('    V  in [%.1f, %.1f] V  : %d/%d\n', p.V_pack_min, p.V_pack_max, ...
        sum(V_means >= p.V_pack_min & V_means <= p.V_pack_max), n);
    fprintf('    IR in [0, 200] mOhm   : %d/%d\n', ...
        sum(IRs*1000 >= 0 & IRs*1000 <= 200), n);

    % =====================================================================
    % EXPORT TABLE
    % =====================================================================
    T_tbl = table( ...
        battID(1:n), cycleNums, ...
        SOC_starts, SOC_ends, DoDs, ...
        IRs, QDs, QCs, ...
        V_means, V_mins, V_maxs, ...
        Tavgs, Tmins, Tmaxs, ...
        chargetimes, discharge_times, ...
        sohs, T_ambs, FAULT, ...
        'VariableNames', { ...
            'battery_id', 'cycle', ...
            'SOC_start', 'SOC_end', 'DoD', ...
            'IR_ohm', 'QD_Ah', 'QC_Ah', ...
            'V_mean_V', 'V_min_V', 'V_max_V', ...
            'Tavg_C', 'Tmin_C', 'Tmax_C', ...
            'chargetime_min', 'discharge_time_min', ...
            'SOH', 'T_amb_K', 'fault_type'});

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    csv_name  = sprintf('battery_dataset_v3_%s.csv', timestamp);
    writetable(T_tbl, csv_name);

    % =====================================================================
    % SUMMARY
    % =====================================================================
    fprintf('\n  Summary — %s:\n', battery_id);
    fprintf('    Cycles completed      : %d\n',   n);
    fprintf('    Cycle life (80%%)     : %d\n',   cycle_life_80);
    fprintf('    Cycle life (70%%)     : %d\n',   cycle_life_70);
    fprintf('    SOH  :  %.4f → %.4f  (%.2f%% drop)\n', ...
        sohs(1), sohs(end), (sohs(1)-sohs(end))*100);
    fprintf('    IR   :  %.3f → %.3f mΩ\n', IRs(1)*1000, IRs(end)*1000);
    fprintf('    QD   :  %.4f → %.4f Ah\n', QDs(1), QDs(end));
    fprintf('    QC   :  %.4f → %.4f Ah  (range=[%.3f,%.3f])\n', ...
        QCs(1), QCs(end), min(QCs), max(QCs));
    fprintf('    SOC_start : %.4f – %.4f\n', min(SOC_starts), max(SOC_starts));
    fprintf('    SOC_end   : %.4f – %.4f\n', min(SOC_ends),   max(SOC_ends));
    fprintf('    DoD       : %.1f%% – %.1f%%  mean=%.1f%%\n', ...
        min(DoDs)*100, max(DoDs)*100, mean(DoDs)*100);
    fprintf('    discharge_time: %.1f – %.1f min  mean=%.3f  std=%.3f\n', ...
        min(discharge_times), max(discharge_times), ...
        mean(discharge_times), std(discharge_times));
    fprintf('    chargetime   : %.1f – %.1f min  mean=%.3f  std=%.3f\n', ...
        min(chargetimes), max(chargetimes), ...
        mean(chargetimes), std(chargetimes));
    fprintf('    Tavg         : %.3f °C  [%.1f, %.1f]\n', ...
        mean(Tavgs), min(Tmins), max(Tmaxs));
    fprintf('    V_mean       : %.3f V   [%.3f, %.3f]\n', ...
        mean(V_means), min(V_mins), max(V_maxs));
    fprintf('    IR mean      : %.4f Ω  (%.2f mΩ)\n', mean(IRs), mean(IRs)*1000);

    fprintf('\n  Fault distribution:\n');
    ft_unique = unique(FAULT);
    for i = 1:length(ft_unique)
        cnt = sum(strcmp(FAULT, ft_unique{i}));
        fprintf('    %-20s: %4d cycles (%5.1f%%)\n', ft_unique{i}, cnt, 100*cnt/n);
    end

    fprintf('\n  Dataset: %s  (%d rows x %d cols)\n\n', ...
            csv_name, height(T_tbl), width(T_tbl));
    disp(T_tbl(1:5, {'cycle','SOC_start','SOC_end','DoD', ...
                     'QD_Ah','QC_Ah','IR_ohm','discharge_time_min', ...
                     'chargetime_min','SOH','fault_type'}));

end  % battery loop

fprintf('=====================================================\n');
fprintf('Simulation complete.\n');
fprintf('=====================================================\n');

% =========================================================================
% HELPER
% =========================================================================
function chosen = sample_fault(scenarios)
    r = rand() * 100; cum = 0; chosen = scenarios{1,1};
    for i = 1:size(scenarios,1)
        cum = cum + scenarios{i,2};
        if r <= cum, chosen = scenarios{i,1}; return; end
    end
end