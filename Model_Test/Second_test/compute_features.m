function feat = compute_features(cycle, row, history)
% === EXACT MATCH to feature_engineering.py ===

feat = struct();

ROLLING_WINDOW = 10;
COLD_THRESHOLD_C = -5;
ECLIPSE_THRESHOLD_K = 268;

% 1-5 Voltage
feat.V_mean_V       = row.V_mean_V;
feat.V_spread       = row.V_max_V - row.V_min_V;
feat.V_mean_rolling = mean(history.V_mean_V(max(1,end-ROLLING_WINDOW+1):end));
feat.V_mean_lag_1   = history.V_mean_V(end);           % last value = lag1
feat.V_min_fade     = row.V_min_V - history.V_min_V(1);

% 6-11 Temperature
feat.Tavg_C         = row.Tavg_C;
feat.thermal_range  = row.Tmax_C - row.Tmin_C;
feat.delta_T_ambient = row.Tavg_C - (row.T_amb_K - 273.15);
feat.cold_cycle_count = sum(history.Tmin_C < COLD_THRESHOLD_C);
feat.Tavg_rolling   = mean(history.Tavg_C(max(1,end-ROLLING_WINDOW+1):end));
feat.eclipse_flag   = double(row.T_amb_K < ECLIPSE_THRESHOLD_K);

% 12-16 Current/Capacity
feat.QD_Ah          = row.QD_Ah;
feat.coulombic_eff  = row.QD_Ah / max(row.QC_Ah, 1e-6);
feat.discharge_C_rate = row.QD_Ah / (row.discharge_time_min/60);
feat.capacity_retention = row.QD_Ah / history.QD_Ah(1);
feat.QD_rolling     = mean(history.QD_Ah(max(1,end-ROLLING_WINDOW+1):end));

% 17-20 OBC
feat.cycle          = cycle;
feat.cumul_Ah       = sum(history.QD_Ah);
feat.SOH_lag_1      = history.SOH_lag_1(end);
feat.QD_diff        = row.QD_Ah - history.QD_Ah(end);

end