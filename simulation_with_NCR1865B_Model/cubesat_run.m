% =========================================================================

clear; close all; clc

p      = cubesat_params();
N_max  = 7000;
n_batt = 1;

battery_ids = {};
for b = 1:n_batt
    battery_ids{b} = sprintf('BATT_EO_%03d', b);
end

fprintf('=====================================================\n');
fprintf('Battery Cycle Dataset Generator  -  NORMAL ONLY\n');
fprintf('Cell model : %s  Pack: %s  Chemistry: %s\n', ...
        p.cell_model, p.pack_config, p.chemistry);
fprintf('=====================================================\n');
fprintf('Q_nom   = %.3f Ah   R0 = %.2f mOhm   k_R = %.3f\n', ...
        p.Q_nom, p.R0*1000, p.k_R);
fprintf('alpha   = %.5f    I_dch = %.2f A   I_ch = %.4f A\n', ...
        p.alpha, p.I_dch, p.I_ch);
fprintf('C-rate (dch) = %.3fC   C-rate (ch) = %.3fC\n', ...
        p.I_dch/p.Q_nom, p.I_ch/p.Q_nom);
fprintf('QD = SOH x Q_nom   QC = SOH x Q_nom x eta\n');
fprintf('SOH_eol = %.2f  - simulation stops when SOH < SOH_eol\n', p.SOH_eol);
fprintf('=====================================================\n\n');

% =========================================================================
for batt_idx = 1:n_batt

    battery_id = battery_ids{batt_idx};
    rng(batt_idx * 42);

    n = N_max;

    % --- Preallocate ---
    battID          = repmat({battery_id}, n, 1);
    cycleNums       = (1:n)';
    SOC_starts      = zeros(n,1);
    SOC_ends        = zeros(n,1);
    DoDs            = zeros(n,1);
    IRs             = zeros(n,1);
    QDs             = zeros(n,1);
    QCs             = zeros(n,1);
    V_means         = zeros(n,1);
    V_mins          = zeros(n,1);
    V_maxs          = zeros(n,1);
    Tavgs           = zeros(n,1);
    Tmins           = zeros(n,1);
    Tmaxs           = zeros(n,1);
    chargetimes     = zeros(n,1);
    discharge_times = zeros(n,1);
    sohs            = zeros(n,1);
    T_ambs          = zeros(n,1);

    % Initial state: SOC=0.99, T=-5 deg C above floor, SOH=1.0
    x = [0.99; p.T_operating_min + 5.0; 1.0];

    fprintf('Battery %s:\n', battery_id);

    for cycle = 1:n

        % =================================================================
        % PER-CYCLE ORBIT TIMES
        % eclipse_min : clipped Gaussian N(35, 1²) in [32, 38] min  — unchanged
        % sunlight_min: clipped Gaussian N(55, 0.8²) in [53, 57] min — CHANGED
        %               (was hardcoded 55.0; now varies each cycle)
        %               I_ch is NOT recomputed; only the available window varies.
        % =================================================================
        eclipse_min  = max(32.0, min(38.0, 35.0 + 1.0  * randn()));
        sunlight_min = max(53.0, min(57.0, 55.0 + 0.8  * randn()));  % <-- CHANGED

        p_cycle = p;
        p_cycle.eclipse_time  = eclipse_min  * 60;
        p_cycle.sunlight_time = sunlight_min * 60;

        % Alternate eclipse structure temperature
        if mod(cycle, 2) == 1
            p_cycle.T_amb_eclipse = p.T_operating_min;        % -10 deg C
        else
            p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;  %  -2 deg C
        end

        % =================================================================
        % SIMULATE ONE ORBIT
        % =================================================================
        row = simulate_cycle(x, p_cycle, 'normal', cycle);

        % =================================================================
        % STORE
        % =================================================================
        SOC_starts(cycle)      = row.SOC_start;
        SOC_ends(cycle)        = row.SOC_end;
        DoDs(cycle)            = row.DoD;
        IRs(cycle)             = row.IR_ohm;
        QDs(cycle)             = row.QD_Ah;
        QCs(cycle)             = row.QC_Ah;
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

        x = row.x_next;

        % Progress print
        if mod(cycle, 200) == 0 || cycle == 1 || cycle == n
            fprintf(['  Cycle %4d/%d | SOH=%.4f | ' ...
                     'SOC %.3f->%.3f (DoD=%.1f%%) | ' ...
                     'QD=%.3fAh QC=%.3fAh | ' ...
                     'IR=%.1fmO | t_d=%.1fm t_c=%.3fm\n'], ...
                cycle, n, sohs(cycle), ...
                SOC_starts(cycle), SOC_ends(cycle), DoDs(cycle)*100, ...
                QDs(cycle), QCs(cycle), IRs(cycle)*1000, ...
                discharge_times(cycle), chargetimes(cycle));
        end

        % =================================================================
        % SOH END-OF-LIFE STOP
        % =================================================================
        if sohs(cycle) < p.SOH_eol
            fprintf('\n  *** SOH = %.4f  <  SOH_eol = %.2f ***\n', ...
                    sohs(cycle), p.SOH_eol);
            fprintf('  Battery end-of-life at cycle %d. Stopping.\n\n', cycle);
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
    fprintf('    Cycles completed         : %d/%d\n', n, N_max);
    fprintf('    Finite (QD, QC, IR)      : %d/%d\n', ...
        sum(isfinite(QDs)&isfinite(QCs)&isfinite(IRs)), n);
    fprintf('    QD in [%.2f, %.2f] Ah   : %d/%d\n', ...
        p.Q_nom*p.SOH_min, p.Q_nom, ...
        sum(QDs >= p.Q_nom*p.SOH_min & QDs <= p.Q_nom*1.01), n);
    fprintf('    QC in [2.30, 3.40] Ah   : %d/%d\n', ...
        sum(QCs >= 2.30 & QCs <= 3.40), n);
    fprintf('    DoD = SOC_s - SOC_e      : %d/%d consistent\n', ...
        sum(abs(SOC_starts-SOC_ends-DoDs) < 1e-5), n);
    fprintf('    T  in [-10, +50] deg C   : %d/%d\n', ...
        sum(Tavgs >= -10 & Tavgs <= 50), n);
    fprintf('    V  in [%.1f, %.1f] V     : %d/%d\n', p.V_pack_min, p.V_pack_max, ...
        sum(V_means >= p.V_pack_min & V_means <= p.V_pack_max), n);
    fprintf('    IR in [0, 300] mOhm      : %d/%d\n', ...
        sum(IRs*1000 >= 0 & IRs*1000 <= 300), n);
    fprintf('    chargetime range (min)   : %.3f - %.3f\n', ...
        min(chargetimes), max(chargetimes));

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
        sohs, T_ambs, ...
        'VariableNames', { ...
            'battery_id', 'cycle', ...
            'SOC_start', 'SOC_end', 'DoD', ...
            'IR_ohm', 'QD_Ah', 'QC_Ah', ...
            'V_mean_V', 'V_min_V', 'V_max_V', ...
            'Tavg_C', 'Tmin_C', 'Tmax_C', ...
            'chargetime_min', 'discharge_time_min', ...
            'SOH', 'T_amb_K'});

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    csv_name  = sprintf('battery_dataset_%s_%s_%s.csv', ...
                        strrep(p.cell_model,' ','_'), p.pack_config, timestamp);
    writetable(T_tbl, csv_name);

    % =====================================================================
    % SUMMARY
    % =====================================================================
    fprintf('\n  Summary -- %s  (%s  %s  %s):\n', ...
            battery_id, p.cell_model, p.pack_config, p.chemistry);
    fprintf('    Cycles completed         : %d\n',   n);
    fprintf('    EOL reached (SOH<%.0f%%)  : %s\n', p.SOH_eol*100, ...
            string(sohs(end) < p.SOH_eol));
    fprintf('    Cycle life (80%%)        : %d\n',   cycle_life_80);
    fprintf('    Cycle life (70%%)        : %d\n',   cycle_life_70);
    fprintf('    SOH   : %.4f -> %.4f  (%.2f%% drop)\n', ...
        sohs(1), sohs(end), (sohs(1)-sohs(end))*100);
    fprintf('    IR    : %.3f -> %.3f mOhm\n', IRs(1)*1000, IRs(end)*1000);
    fprintf('    QD    : %.4f -> %.4f Ah\n', QDs(1), QDs(end));
    fprintf('    QC    : %.4f -> %.4f Ah  range=[%.3f, %.3f]\n', ...
        QCs(1), QCs(end), min(QCs), max(QCs));
    fprintf('    SOC_start : %.4f - %.4f\n', min(SOC_starts), max(SOC_starts));
    fprintf('    SOC_end   : %.4f - %.4f\n', min(SOC_ends),   max(SOC_ends));
    fprintf('    DoD       : %.1f%% - %.1f%%  mean=%.1f%%\n', ...
        min(DoDs)*100, max(DoDs)*100, mean(DoDs)*100);
    fprintf('    discharge_time : %.3f - %.3f min  mean=%.3f\n', ...
        min(discharge_times), max(discharge_times), mean(discharge_times));
    fprintf('    chargetime     : %.3f - %.3f min  mean=%.3f\n', ...
        min(chargetimes), max(chargetimes), mean(chargetimes));
    fprintf('    Tavg  : %.3f C   [%.1f, %.1f]\n', ...
        mean(Tavgs), min(Tmins), max(Tmaxs));
    fprintf('    V_mean: %.3f V   [%.3f, %.3f]\n', ...
        mean(V_means), min(V_mins), max(V_maxs));
    fprintf('    IR mean: %.3f mOhm\n', mean(IRs)*1000);

    fprintf('\n  Dataset saved: %s  (%d rows x %d cols)\n\n', ...
            csv_name, height(T_tbl), width(T_tbl));
    disp(T_tbl(1:min(5,n), {'cycle','SOC_start','SOC_end','DoD', ...
                     'QD_Ah','QC_Ah','IR_ohm','discharge_time_min', ...
                     'chargetime_min','SOH'}));
end

fprintf('=====================================================\n');
fprintf('Simulation complete.\n');
fprintf('=====================================================\n');
