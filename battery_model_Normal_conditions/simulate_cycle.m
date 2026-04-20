function row = simulate_cycle(x0, p, cycle_num)
% =========================================================================

if nargin < 4 || isempty(cycle_num)
    cycle_num = 1;
end

% =========================================================================
% PHASE 1 — DISCHARGE (eclipse)
% =========================================================================
SOC_start = x0(1);

opts_d = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'MaxStep', 10, ...
                'NonNegative', [1,2,3], ...
                'Events', @(t,x) evt_dch(t, x, p));

[t_d, X_d] = ode45(@(t,x) cubesat_ode_dch(t, x, p), ...
                   [0, p.eclipse_time], x0, opts_d);

IR_d = zeros(length(t_d), 1);
V_d  = zeros(length(t_d), 1);
for k = 1:length(t_d)
    [~, V_d(k), IR_d(k)] = cubesat_ode_dch(t_d(k), X_d(k,:)', p);
end

actual_dch_s = t_d(end);
% SOC_end = SOC after DISCHARGE (depleted state, not after charge)
SOC_end = X_d(end, 1);
DoD     = SOC_start - SOC_end;
T_d_C   = X_d(:, 2) - 273.15;

% =========================================================================
% PHASE 2 — CHARGE (sunlight)
% =========================================================================
opts_c = odeset(opts_d, 'Events', @(t,x) evt_ch(t, x));

[t_c, X_c] = ode45(@(t,x) cubesat_ode_ch(t, x, p), ...
                   [0, p.sunlight_time], X_d(end,:)', opts_c);

if length(t_c) >= 2
    T_c_C       = X_c(:, 2) - 273.15;
    actual_ch_s = t_c(end);
    V_c = zeros(length(t_c), 1);
    for k = 1:length(t_c)
        [~, V_c(k), ~] = cubesat_ode_ch(t_c(k), X_c(k,:)', p);
    end
    x_next = X_c(end, :)';
else
    T_c_C       = T_d_C;
    actual_ch_s = 0;
    V_c         = V_d;
    x_next      = X_d(end, :)';
    warning('simulate_cycle: cycle %d charge phase < 2 points.', cycle_num);
end

% =========================================================================
% CAPACITY FEATURES  (v3 — key fix)
% =========================================================================

SOH_now = x_next(3);   % SOH at end of this cycle (after both phases)
QD = SOH_now * p.Q_nom;            % Ah  rated discharge capacity
QC = SOH_now * p.Q_nom * p.eta;    % Ah  rated charge capacity

% =========================================================================
% FEATURE EXTRACTION
% =========================================================================
T_all = [T_d_C; T_c_C];
V_all = [V_d;   V_c  ];
T_lo  = p.T_operating_min - 273.15;   % −10 °C
T_hi  = p.T_operating_max - 273.15;   % +50 °C

T_avg_C = max(T_lo, min(T_hi, mean(T_all)));
T_min_C = max(T_lo, min(T_hi, min(T_all)));
T_max_C = max(T_lo, min(T_hi, max(T_all)));

% =========================================================================
% VALIDATION
% =========================================================================
if ~isfinite(QD) || ~isfinite(QC) || QD > p.Q_nom * 1.1 || QD < 0
    warning('simulate_cycle: cycle %d — unexpected QD=%.4f Ah.', cycle_num, QD);
end

% =========================================================================
% OUTPUT ROW
% =========================================================================
row.SOC_start          = SOC_start;
row.SOC_end            = SOC_end;          % SOC after DISCHARGE
row.DoD                = DoD;
row.IR_ohm             = mean(IR_d);       % mean pack resistance (discharge)
row.QD_Ah              = QD;               % rated capacity (SOH * Q_nom)
row.QC_Ah              = QC;               % rated charge cap (SOH * Q_nom * eta)
row.V_mean_V           = mean(V_all);
row.V_min_V            = min(V_all);
row.V_max_V            = max(V_all);
row.Tavg_C             = T_avg_C;
row.Tmin_C             = T_min_C;
row.Tmax_C             = T_max_C;
row.chargetime_min     = actual_ch_s / 60;         % ACTUAL solver time
row.discharge_time_min = actual_dch_s / 60;        % ACTUAL solver time
row.x_next             = x_next;
row.SOH_end            = x_next(3);
row.fault_type         = fault_type;
row.T_amb_K            = p.T_amb_eclipse;

end

% =========================================================================
% LOCAL EVENT FUNCTIONS
% =========================================================================

function [val, isterminal, direction] = evt_dch(~, x, p)
    val        = x(1) - p.SOC_min;
    isterminal = 1;
    direction  = -1;
end

function [val, isterminal, direction] = evt_ch(~, x)
    val        = x(1) - 0.99;
    isterminal = 1;
    direction  = +1;
end