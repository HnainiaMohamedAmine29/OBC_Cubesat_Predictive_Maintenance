function feat = compute_features(cycle, row, history)

feat = struct();
W = 10; % must match training ROLLING_W

%% =========================
%% Helper indices (rolling like pandas)
%% =========================
idx_start = max(1, cycle - W + 1);
idx_range = idx_start:cycle;

%% =========================
%% Voltage (MATCH pandas exactly)
%% =========================
feat.V_mean_V = row.V_mean_V;
feat.V_spread = row.V_max_V - row.V_min_V;

feat.V_mean_rolling = mean(history.V_mean_V(idx_range));

if cycle > 1
    feat.V_mean_lag_1 = history.V_mean_V(cycle-1);
else
    feat.V_mean_lag_1 = 0;
end

feat.V_min_fade = row.V_min_V - history.V_min_V(1);

%% =========================
%% Temperature
%% =========================
feat.Tavg_C = row.Tavg_C;
feat.thermal_range = row.Tmax_C - row.Tmin_C;

if cycle > 1
    feat.delta_T_ambient = row.T_amb_K - history.T_amb_K(cycle-1);
else
    feat.delta_T_ambient = 0;
end

feat.cold_cycle_count = sum(history.Tavg_C(1:cycle) < 5);
feat.Tavg_rolling = mean(history.Tavg_C(idx_range));

feat.eclipse_flag = double(row.T_amb_K <= 263.15);

%% =========================
%% Capacity
%% =========================
feat.cycle = cycle;   % ✅ FIX ADDED

feat.QD_Ah = row.QD_Ah;

feat.coulombic_eff = row.QC_Ah / max(row.QD_Ah, 1e-6);

feat.discharge_C_rate = row.DoD / max(row.discharge_time_min/60, 1e-6);

feat.capacity_retention = row.QD_Ah / max(history.QD_Ah(1), 1e-6);

feat.QD_rolling = mean(history.QD_Ah(idx_range));

feat.cumul_Ah = sum(history.QD_Ah(1:cycle));

if cycle > 1
    feat.QD_diff = row.QD_Ah - history.QD_Ah(cycle-1);
else
    feat.QD_diff = 0;
end

%% =========================
%% SOH lag (GROUND TRUTH ONLY)
%% =========================
if isfield(history, 'SOH')
    if cycle > 1
        feat.SOH_lag_1 = history.SOH(cycle-1);
    else
        feat.SOH_lag_1 = 1.0;
    end
else
    feat.SOH_lag_1 = 1.0;
end

end