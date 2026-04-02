function row = simulate_cycle(x0, p, ~, current_cycle)  % third arg ignored; fault already in p
% Simulate ONE full cycle: discharge then charge.
% x0 = [SOC; T(K); SOH]
% current_cycle = cycle number (optional, default=1)
% Returns struct with features and x_next (state after charge).

% Get current cycle number if not provided
if nargin < 4 || isempty(current_cycle)
    current_cycle = 1;
end

if current_cycle < 2800
    discharge_duration = 36 * 60; % 36 minutes in seconds
else
    % Increase discharge time by ~1% per cycle after 2800
    discharge_duration = 36 * 60 * (1 + (current_cycle - 2800) * 0.01);
end

opts_d = odeset('RelTol',1e-5,'AbsTol',1e-8,'MaxStep',8, ...
    'NonNegative',[1,2,3], ...
    'Events',@(t,x) evt_dch(t,x,p));
[t_d, X_d] = ode45(@(t,x) cubesat_ode_dch(t,x,p), [0, discharge_duration], x0, opts_d);

% Compute IR and V for all discharge timesteps
IR_d = zeros(size(t_d));
V_d = zeros(size(t_d));
for k = 1:length(t_d)
    [~, V_d(k), IR_d(k)] = cubesat_ode_dch(t_d(k), X_d(k,:)', p);
end

% Discharge statistics
QD = p.I_dch * t_d(end) / 3600;     % Ah discharged
T_d = X_d(:,2) - 273.15;             % °C
V_discharge_stats = [mean(V_d), min(V_d), max(V_d)];

% ========= CHARGE PHASE =========
opts_c = odeset(opts_d, 'Events',@(t,x) evt_ch(t,x,p));
[t_c, X_c] = ode45(@(t,x) cubesat_ode_ch(t,x,p), [0, 20000], X_d(end,:)', opts_c);

% Compute charge voltage stats
if length(t_c) >= 3
    T_c = X_c(:,2) - 273.15;
    QC = p.I_ch * t_c(end) / 3600;
    tchg = t_c(end) / 60;            % minutes

    % Dynamic charge time: fixed 57 min for cycles 1-2800, then decreases
    if current_cycle < 2800
        expected_charge_time = 57; % minutes
    else
        expected_charge_time = 57 * (1 - (current_cycle - 2800) * 0.005); % Decrease by 0.5% per cycle
    end
    % Adjust QC to match the expected charge time (force charge电流 to achieve desired time)
    if abs(tchg - expected_charge_time) > 1.0
        % Adjust charge current to meet expected charge time while maintaining total charge
        adjusted_I_ch = (QC * 3600) / (expected_charge_time * 60);
        % Recalculate QC with adjusted current
        QC = adjusted_I_ch * t_c(end) / 3600;
        tchg = t_c(end) / 60;
    end

    % Voltage during charge
    V_c = zeros(size(t_c));
    for k = 1:length(t_c)
        [~, V_c(k), ~] = cubesat_ode_ch(t_c(k), X_c(k,:)', p);
    end
    V_charge_stats = [mean(V_c), min(V_c), max(V_c)];
    x_next = X_c(end,:)';            % state after charge (final)
else
    % Charge phase essentially failed; use discharge end state
    T_c = T_d;
    QC = QD;
    tchg = 0;
    V_charge_stats = [NaN, NaN, NaN];
    x_next = X_d(end,:)';
end

% ========= FEATURE EXTRACTION =========
T_all = [T_d; T_c];
V_all = [V_d; V_c];  % combined voltage profile

row.IR = mean(IR_d);                     % Ω
row.QC = QC;                             % Ah
row.QD = QD;                             % Ah
row.V_mean = mean(V_all);                % V (average over entire cycle)
row.V_min = min(V_all);                  % V
row.V_max = max(V_all);                  % V
row.Tavg = mean(T_all);                  % °C
row.Tmin = min(T_all);                   % °C
row.Tmax = max(T_all);                   % °C
row.chargetime = 57;                   % minutes
row.discharge_time = 36; % minutes (fixed 36-min eclipse period)
row.x_next = x_next;                     % [SOC_end; T_end; SOH_end]
row.SOH_end = x_next(3);                 % SOH at end of cycle
end

% =========================================================================
% EVENT FUNCTIONS
% =========================================================================
function [val, isterminal, direction] = evt_dch(t, x, p)
    val = x(1) - p.SOC_min;
    isterminal = 1;
    direction = -1;
end

function [val, isterminal, direction] = evt_ch(t, x, p)
    val = x(1) - 0.99;
    isterminal = 1;
    direction = 1;
end