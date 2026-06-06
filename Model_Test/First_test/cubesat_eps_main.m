%% ================================================
%% CubeSat Real-Time Simulation + LSTM SOH Prediction
%% Raw data → compute_features.m → LSTM
%% ================================================

clear; close all; clc

p = cubesat_params();

fprintf('=== CubeSat Real-Time LSTM SOH Prediction ===\n');
fprintf('Raw features → Feature Engineering → LSTM\n\n');

% Initialize Python LSTM Predictor
pyenv;
predictor = py.importlib.import_module('lstm_core');
py_predictor = predictor.CubeSatSOHPredictor();

% Preallocate
n_max = 7000;
cycleNums = (1:n_max)';

SOH_true = zeros(n_max,1);
SOH_lstm = zeros(n_max,1);

% History for feature engineering (raw data)
history = struct( ...
    'V_mean_V', [], ...
    'V_min_V',  [], ...
    'V_max_V',  [], ...
    'Tavg_C',   [], ...
    'Tmin_C',   [], ...
    'Tmax_C',   [], ...
    'QD_Ah',    [], ...
    'QC_Ah',    [], ...
    'T_amb_K',  [], ...
    'discharge_time_min', [], ...
    'SOH_lag_1', [] ...
);

x = [0.99; p.T_operating_min + 5.0; 1.0];
soh_prev = 1.0;

fprintf('Simulation started...\n');

for cycle = 1:n_max
    % === 1. Simulate one cycle (raw features) ===
    row = simulate_cycle(x, p, cycle);
    
    % Store true SOH
    SOH_true(cycle) = row.SOH_end;
    
    % === 2. Append raw data to history ===
    history.V_mean_V(end+1)      = row.V_mean_V;
    history.V_min_V(end+1)       = row.V_min_V;
    history.V_max_V(end+1)       = row.V_max_V;
    history.Tavg_C(end+1)        = row.Tavg_C;
    history.Tmin_C(end+1)        = row.Tmin_C;
    history.Tmax_C(end+1)        = row.Tmax_C;
    history.QD_Ah(end+1)         = row.QD_Ah;
    history.QC_Ah(end+1)         = row.QC_Ah;
    history.T_amb_K(end+1)       = row.T_amb_K;
    history.discharge_time_min(end+1) = row.discharge_time_min;
    history.SOH_lag_1(end+1)     = soh_prev;
    
    x = row.x_next;
    
    % === 3. Feature Engineering using dedicated function ===
    feat_struct = compute_features(cycle, row, history);
    feature_dict = py.dict(feat_struct);
    
    % === 4. LSTM Prediction ===
    if cycle >= 10
        SOH_lstm(cycle) = double(py_predictor.predict_soh(feature_dict, soh_prev));
        soh_prev = SOH_lstm(cycle);
        history.SOH_lag_1(end) = SOH_lstm(cycle);   % recursive
    else
        SOH_lstm(cycle) = SOH_true(cycle);
    end
    
    % Progress
    if mod(cycle, 100) == 0 || cycle == 1 || SOH_true(cycle) < 0.72
        fprintf('Cycle %4d | True SOH: %.4f | LSTM SOH: %.4f | QD: %.3f Ah\n', ...
            cycle, SOH_true(cycle), SOH_lstm(cycle), row.QD_Ah);
    end
    
    % Stop at EOL
    if SOH_true(cycle) < 0.70
        fprintf('\n🔴 END-OF-LIFE REACHED at cycle %d (SOH = %.4f)\n', cycle, SOH_true(cycle));
        break;
    end
end

% Trim data
idx = 1:cycle;
cycleNums = cycleNums(idx);
SOH_true = SOH_true(idx);
SOH_lstm = SOH_lstm(idx);

% Save results
T_result = table(cycleNums, SOH_true, SOH_lstm, ...
    'VariableNames', {'cycle','SOH_true','SOH_lstm'});
writetable(T_result, 'CubeSat_LSTM_Results.csv');
fprintf('✅ Results saved to CubeSat_LSTM_Results.csv\n');

% Plot
figure('Position',[100 100 1000 600]);
plot(cycleNums, SOH_true, 'b-', 'LineWidth', 2.5); hold on;
plot(cycleNums, SOH_lstm, 'r--', 'LineWidth', 1.8);
yline(0.70, 'k--', 'EOL Threshold', 'LineWidth', 1.5);
legend('True SOH (Physics)', 'LSTM Predicted SOH', 'Location','best');
xlabel('Cycle Number'); ylabel('State of Health');
title('CubeSat Battery - Real-Time LSTM SOH Prediction');
grid on;

fprintf('\n🎉 Simulation completed in %d cycles!\n', cycle);