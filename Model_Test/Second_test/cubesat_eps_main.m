%% ================================================
%% CubeSat Normal Battery + LSTM SOH Prediction
%% Clean & Stable Version
%% ================================================
clear; close all; clc;

p = cubesat_params();

fprintf('=== CubeSat Normal Battery + Real-Time LSTM SOH ===\n\n');

% Initialize Python LSTM Predictor
pyenv('ExecutionMode', 'OutOfProcess');   % More stable
predictor = py.importlib.import_module('lstm_core_3');
py.importlib.reload(predictor);
py_predictor = predictor.CubeSatSOHPredict();

fprintf('✅ LSTM Predictor loaded successfully!\n');

% Preallocation
n_max = 8000;
cycleNums = (1:n_max)';

SOH_true = zeros(n_max,1);
SOH_lstm = zeros(n_max,1);

% History for compute_features.m
history = struct();
history.V_mean_V = zeros(n_max,1);
history.V_min_V  = zeros(n_max,1);
history.V_max_V  = zeros(n_max,1);
history.Tavg_C   = zeros(n_max,1);
history.Tmin_C   = zeros(n_max,1);
history.Tmax_C   = zeros(n_max,1);
history.QD_Ah    = zeros(n_max,1);
history.QC_Ah    = zeros(n_max,1);
history.IR_ohm   = zeros(n_max,1);
history.DoD      = zeros(n_max,1);
history.discharge_time_min = zeros(n_max,1);
history.T_amb_K  = zeros(n_max,1);
history.SOH_lag_1 = zeros(n_max,1);

% Initial battery state
x = [0.99; p.T_operating_min + 5.0; 1.0];
soh_prev = 1.0;

fprintf('Starting Normal Battery Simulation...\n');

for cycle = 1:n_max
    % === Simulate one cycle (Normal - No Fault) ===
    row = simulate_cycle(x, p,cycle);   % Explicit 'normal'
    
    SOH_true(cycle) = row.SOH_end;
    
    % Update history (raw features)
    history.V_mean_V(cycle) = row.V_mean_V;
    history.V_min_V(cycle)  = row.V_min_V;
    history.V_max_V(cycle)  = row.V_max_V;
    history.Tavg_C(cycle)   = row.Tavg_C;
    history.Tmin_C(cycle)   = row.Tmin_C;
    history.Tmax_C(cycle)   = row.Tmax_C;
    history.QD_Ah(cycle)    = row.QD_Ah;
    history.QC_Ah(cycle)    = row.QC_Ah;
    history.IR_ohm(cycle)   = row.IR_ohm;
    history.DoD(cycle)      = row.DoD;
    history.discharge_time_min(cycle) = row.discharge_time_min;
    history.T_amb_K(cycle)  = row.T_amb_K;
    history.SOH_lag_1(cycle) = soh_prev;
    
    x = row.x_next;
    
    % === Feature Engineering ===
    feat_struct = compute_features(cycle, row, history);
    
    % === LSTM Prediction ===
    keys = fieldnames(feat_struct);
    values = struct2cell(feat_struct);

    py_dict = py.dict();
    for i = 1:numel(keys)
        py_dict{keys{i}} = values{i};
    end

    feature_dict = py_dict;
    SOH_lstm(cycle) = double(py_predictor.predict_soh(feature_dict, soh_prev));
    soh_prev = SOH_lstm(cycle);
    
    % Progress
    if mod(cycle,100)==0 || cycle==1 || SOH_true(cycle)<0.75
        fprintf('Cycle %4d | True SOH: %.4f | LSTM SOH: %.4f | QD: %.3f Ah\n', ...
            cycle, SOH_true(cycle), SOH_lstm(cycle), row.QD_Ah);
    end
    
    % Stop at 70% SOH
    if SOH_true(cycle) < 0.70
        fprintf('\n🔴 END-OF-LIFE REACHED at cycle %d (SOH = %.4f)\n', cycle, SOH_true(cycle));
        break;
    end
end

% Trim data
idx = 1:cycle;
disp(fieldnames(feat_struct))
% Save Results
T_result = table(cycleNums(idx), SOH_true(idx), SOH_lstm(idx), ...
    'VariableNames', {'cycle','SOH_true','SOH_lstm'});
writetable(T_result, 'CubeSat_Normal_LSTM_Results.csv');

fprintf('✅ Simulation completed in %d cycles!\n', cycle);

% Plot
figure('Position',[100 100 1000 600]);
plot(cycleNums(idx), SOH_true(idx), 'b-', 'LineWidth', 2.5); hold on;
plot(cycleNums(idx), SOH_lstm(idx), 'r--', 'LineWidth', 2);
yline(0.70, 'k--', 'EOL Threshold (70%)', 'LineWidth', 1.5);
legend('True SOH (Physics Model)', 'LSTM Predicted SOH', 'Location','best');
title('CubeSat Battery - Normal Condition LSTM SOH Prediction');
xlabel('Cycle Number'); ylabel('State of Health');
grid on;