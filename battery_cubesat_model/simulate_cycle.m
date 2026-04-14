function row = simulate_cycle(x0, p, fault_type, cycle_num)
% =========================================================================
% Simulate ONE full battery cycle: discharge (eclipse) then charge (sunlight)
% x0 = [SOC; T(K); SOH]
% fault_type = fault label (returned in output)
% cycle_num = cycle number (optional)
%
% CORRECTED:
%  - REMOVED the zero-reaching charge-time formula (caused QC infinity)
%  - Export ACTUAL solver times, not hardcoded values
%  - Add SOC_start, SOC_end, DoD features
%  - All features stay finite and physically bounded
% =========================================================================

if nargin < 4 || isempty(cycle_num)
    cycle_num = 1;
end

% --- DISCHARGE PHASE (eclipse) ---
% Fixed discharge time based on realistic LEO eclipse (35 minutes)
% No arbitrary post-2800 scaling
discharge_duration = p.eclipse_time;    % seconds (already ~2100 s = 35 min from params)

% Record initial SOC before discharge
SOC_start = x0(1);

opts_d = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'MaxStep', 8, ...
    'NonNegative', [1, 2, 3], ...
    'Events', @(t,x) evt_dch(t,x,p));

[t_d, X_d] = ode45(@(t,x) cubesat_ode_dch(t,x,p), [0, discharge_duration], x0, opts_d);

% Compute IR and V for all discharge timesteps
IR_d = zeros(size(t_d));
V_d = zeros(size(t_d));
for k = 1:length(t_d)
    [~, V_d(k), IR_d(k)] = cubesat_ode_dch(t_d(k), X_d(k,:)', p);
end

% Discharge statistics
actual_discharge_time = t_d(end);       % Actual elapsed time (seconds)
QD = p.I_dch * actual_discharge_time / 3600;  % Charge discharged (Ah)
T_d = X_d(:, 2) - 273.15;               % Temperature in Celsius
V_discharge_stats = [mean(V_d), min(V_d), max(V_d)];

% SOC after discharge
SOC_after_discharge = X_d(end, 1);

% --- CHARGE PHASE (sunlight) ---
% Fixed charge time based on realistic LEO sunlight (58 minutes)
% No arbitrary time formula; just let the solver run until SOC reaches limit
charge_duration_max = p.sunlight_time;  % Max time for charge (~3480 s = 58 min)

opts_c = odeset(opts_d, ...
    'Events', @(t,x) evt_ch(t,x,p));

[t_c, X_c] = ode45(@(t,x) cubesat_ode_ch(t,x,p), [0, charge_duration_max], X_d(end,:)', opts_c);

% Compute charge voltage stats
if length(t_c) >= 3
    T_c = X_c(:, 2) - 273.15;
    actual_charge_time = t_c(end);      % ACTUAL solver time (seconds), not arbitrary formula
    QC = p.I_ch * actual_charge_time / 3600;  % Charge gained (Ah), from actual time
    
    % Voltage during charge
    V_c = zeros(size(t_c));
    for k = 1:length(t_c)
        [~, V_c(k), ~] = cubesat_ode_ch(t_c(k), X_c(k,:)', p);
    end
    V_charge_stats = [mean(V_c), min(V_c), max(V_c)];
    x_next = X_c(end, :)';              % Final state [SOC_end; T_end; SOH_end]
    SOC_end = X_c(end, 1);              % SOC after full cycle
else
    % Charge phase ended prematurely; use discharge end state
    T_c = T_d;
    actual_charge_time = t_c(end);
    QC = 0;
    V_charge_stats = [NaN, NaN, NaN];
    x_next = X_d(end, :)';
    SOC_end = X_d(end, 1);
end

% --- FEATURE EXTRACTION & VALIDATION ---
T_all = [T_d; T_c];
V_all = [V_d; V_c];

% Depth of discharge (how much was extracted from battery)
DoD = SOC_start - SOC_after_discharge;

% Charge balance check (optional warning if very unbalanced)
if abs(QC - QD) > 0.5 * max(abs(QC), abs(QD))
    % warning('Cycle %d: Charge/discharge imbalance detected (QC=%.3f, QD=%.3f)', ...
    %         cycle_num, QC, QD);
end

% --- VALIDATION: ensure all features are finite and bounded ---
V_mean_cyc = mean(V_all);
V_min_cyc = min(V_all);
V_max_cyc = max(V_all);
T_avg_cyc = mean(T_all);
T_min_cyc = min(T_all);
T_max_cyc = max(T_all);

% Clip temperatures to managed range (hard limits for export)
T_avg_cyc = max(p.T_operating_min - 273.15, min(p.T_operating_max - 273.15, T_avg_cyc));
T_min_cyc = max(p.T_operating_min - 273.15, min(p.T_operating_max - 273.15, T_min_cyc));
T_max_cyc = max(p.T_operating_min - 273.15, min(p.T_operating_max - 273.15, T_max_cyc));

% --- Build output struct ---
row.SOC_start = SOC_start;
row.SOC_end = SOC_end;
row.DoD = DoD;
row.IR = mean(IR_d);                    % Ω (mean discharge resistance)
row.QC = QC;                            % Ah (from actual charge time)
row.QD = QD;                            % Ah (from actual discharge time)
row.V_mean = V_mean_cyc;                % V
row.V_min = V_min_cyc;                  % V
row.V_max = V_max_cyc;                  % V
row.Tavg = T_avg_cyc;                   % °C
row.Tmin = T_min_cyc;                   % °C
row.Tmax = T_max_cyc;                   % °C
row.chargetime = actual_charge_time / 60;      % minutes (ACTUAL from solver, not hardcoded)
row.discharge_time = actual_discharge_time / 60; % minutes (ACTUAL from solver, not hardcoded)
row.x_next = x_next;                    % [SOC_end; T_end; SOH_end]
row.SOH_end = x_next(3);                % SOH after cycle
row.fault_type = fault_type;            % EXPORT fault label
row.T_amb = p.T_amb_eclipse;            % EXPORT ambient temperature

% --- Sanity check: ensure no infinities or NaN ---
if ~isfinite(row.QC) || ~isfinite(row.QD) || row.QC > 20 || row.QD > 20
    warning('Cycle %d: Invalid values detected (QC=%.3f, QD=%.3f)', cycle_num, row.QC, row.QD);
end

end

% =========================================================================
% EVENT FUNCTIONS (unified, consistent)
% =========================================================================

function [val, isterminal, direction] = evt_dch(t, x, p)
    % Stop discharge when SOC reaches minimum
    val = x(1) - p.SOC_min;
    isterminal = 1;         % Stop integration
    direction = -1;         % Detect decreasing (discharge reduces SOC)
end

function [val, isterminal, direction] = evt_ch(t, x, p)
    % Stop charge when SOC reaches 99%
    val = x(1) - 0.99;
    isterminal = 1;         % Stop integration
    direction = 1;          % Detect increasing (charge increases SOC)
end
