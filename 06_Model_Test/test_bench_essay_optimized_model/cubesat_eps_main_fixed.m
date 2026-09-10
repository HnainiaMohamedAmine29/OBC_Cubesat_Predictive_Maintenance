%% ============================================================
%% CubeSat Battery + REAL-TIME LSTM SOH Prediction (FINAL FIXED)
%% ============================================================

clear; close all; clc;

p = cubesat_params();

rng(42);  % match cubesat_run.m's reproducible seeding

fprintf('=== CubeSat REAL-TIME LSTM SOH Prediction (FINAL FIXED) ===\n\n');

%% ============================================================
%% PYTHON INIT
%% ============================================================
pyenv('Version', 'C:\Users\Amine\Desktop\files _matlab_test_bench\CubeSat_V4_Final\CubeSat_V4\tf_env\Scripts\python.exe', ...
      'ExecutionMode','OutOfProcess');

module = py.importlib.import_module('lstm_core_5');
py.importlib.reload(module);
py_predictor = module.CubeSatSOHPredict();

fprintf('✅ LSTM Predictor Loaded\n');

%% ============================================================
%% CONFIG
%% ============================================================
n_max = 8000;
cycleNums = (1:n_max)';

SOH_true = zeros(n_max,1);
SOH_lstm = nan(n_max,1);

%% ============================================================
%% FEATURE LOGGING (for comparing vs. training raw_features.csv)
%% ============================================================
log_features = {'V_mean_V','V_spread','V_mean_rolling','V_mean_lag_1','V_min_fade', ...
                 'Tavg_C','thermal_range','delta_T_ambient','cold_cycle_count', ...
                 'Tavg_rolling','eclipse_flag','QD_Ah','coulombic_eff', ...
                 'discharge_C_rate','capacity_retention','QD_rolling','cycle', ...
                 'cumul_Ah','SOH_lag_1','QD_diff'};
feature_log = nan(n_max, numel(log_features));

%% ============================================================
%% HISTORY BUFFER
%% ============================================================
history = struct();

fields = {'V_mean_V','V_min_V','V_max_V','Tavg_C','Tmin_C','Tmax_C',...
          'QD_Ah','QC_Ah','IR_ohm','DoD','discharge_time_min','T_amb_K'};

for f = 1:numel(fields)
    history.(fields{f}) = zeros(n_max,1);
end

%% ============================================================
%% INITIAL STATE
%% ============================================================
x = [0.99; p.T_operating_min + 5.0; 1.0];

fprintf('🚀 Starting simulation...\n\n');

%% ============================================================
%% FEATURE ORDER (CRITICAL)
%% ============================================================
ordered_features = { ...
    'V_mean_V', ...
    'V_spread', ...
    'V_mean_rolling', ...
    'V_mean_lag_1', ...
    'V_min_fade', ...
    'Tavg_C', ...
    'thermal_range', ...
    'delta_T_ambient', ...
    'cold_cycle_count', ...
    'Tavg_rolling', ...
    'eclipse_flag', ...
    'QD_Ah', ...
    'coulombic_eff', ...
    'discharge_C_rate', ...
    'capacity_retention', ...
    'QD_rolling', ...
    'cycle', ...
    'cumul_Ah', ...
    'SOH_lag_1', ...
    'QD_diff' ...
};

%% ============================================================
%% MAIN LOOP
%% ============================================================
for cycle = 1:n_max
    
    %% 1️⃣ Build this cycle's params (matches cubesat_run.m exactly)
    eclipse_min  = max(32.0, min(38.0, 35.0 + 1.0 * randn()));
    sunlight_min = max(53.0, min(57.0, 55.0 + 0.8 * randn()));

    p_cycle = p;
    p_cycle.eclipse_time  = eclipse_min  * 60;
    p_cycle.sunlight_time = sunlight_min * 60;

    if mod(cycle, 2) == 1
        p_cycle.T_amb_eclipse = p.T_operating_min;        % -10 C
    else
        p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;  %  -2 C
    end

    %% 2️⃣ Simulate cycle
    row = simulate_cycle(x, p_cycle, cycle);
    SOH_true(cycle) = row.SOH_end;
    
    %% 3️⃣ Update history
    for f = 1:numel(fields)
        history.(fields{f})(cycle) = row.(fields{f});
    end
    
    x = row.x_next;
    
    %% 4️⃣ Feature engineering
    feat_struct = compute_features(cycle, row, history);
    
    %% 🔥 TRUE SOH lag
    if cycle == 1
     feat_struct.SOH_lag_1 = 1.0;
   else
     feat_struct.SOH_lag_1 = SOH_true(cycle-1);
   end

    %% Log every feature value this cycle for later comparison vs. training stats
    for lf = 1:numel(log_features)
        feature_log(cycle, lf) = feat_struct.(log_features{lf});
    end
    
    %% ========================================================
    %% 4️⃣ BUILD PYTHON DICTIONARY (FIXED)
    %% ========================================================
    py_dict = py.dict();

    for i = 1:length(ordered_features)
        key = ordered_features{i};

        if ~isfield(feat_struct, key)
            error("❌ Missing feature: %s", key);
        end

        val = feat_struct.(key);

        % FIX: No NaN / Inf allowed
        if isnan(val) || isinf(val)
            val = 0.0;
        end

        py_dict{key} = val;
    end

    %% ========================================================
    %% 5️⃣ PYTHON PREDICTION (SAFE)
    %% ========================================================
    % try
        soh_pred = py_predictor.predict_soh(py_dict);
    % catch ME
        % warning("Python error at cycle %d: %s", cycle, ME.message);
        % soh_pred = py.None;
    % end

    %% ========================================================
    %% 6️⃣ HANDLE OUTPUT
    %% ========================================================
    if isequal(soh_pred, py.None)
        SOH_lstm(cycle) = NaN;
    else
        SOH_lstm(cycle) = double(soh_pred);
    end

    %% ========================================================
    %% 7️⃣ LOGGING
    %% ========================================================
    if mod(cycle,200)==0 || cycle==1 || SOH_true(cycle)<0.75
        fprintf('Cycle %4d | True SOH: %.4f | Pred SOH: %.4f | QD: %.3f Ah\n', ...
            cycle, SOH_true(cycle), SOH_lstm(cycle), row.QD_Ah);
    end

    %% ========================================================
    %% 8️⃣ END OF LIFE
    %% ========================================================
    if SOH_true(cycle) < 0.70
        fprintf('\n🔴 EOL reached at cycle %d | SOH = %.4f\n', ...
            cycle, SOH_true(cycle));
        break;
    end
end

%% ============================================================
%% RESULTS TABLE
%% ============================================================
idx = 1:cycle;

T_result = table(cycleNums(idx), SOH_true(idx), SOH_lstm(idx), ...
    'VariableNames', {'cycle','SOH_true','SOH_lstm'});

writetable(T_result, 'CubeSat_REALTIME_LSTM_FINAL.csv');

%% ============================================================
%% WRITE FEATURE LOG (for comparison against training raw_features.csv)
%% ============================================================
T_features = array2table(feature_log(idx,:), 'VariableNames', log_features);
T_features.cycle_idx = cycleNums(idx);
writetable(T_features, 'CubeSat_REALTIME_FEATURE_LOG.csv');
fprintf('✅ Feature log written to CubeSat_REALTIME_FEATURE_LOG.csv\n');

fprintf('\n✅ Simulation finished (%d cycles)\n', cycle);

%% ============================================================
%% PLOT RESULTS
%% ============================================================
figure('Position',[100 100 1200 700]);

plot(cycleNums(idx), SOH_true(idx), 'b-', 'LineWidth', 2.5); hold on;

valid = ~isnan(SOH_lstm);
plot(cycleNums(valid), SOH_lstm(valid), 'r--', 'LineWidth', 2.2);

yline(0.70,'k--','EOL 70%');

legend('True SOH','Predicted SOH');
title('CubeSat LSTM Real-Time SOH Prediction (FINAL)');
xlabel('Cycle'); ylabel('SOH');
grid on;