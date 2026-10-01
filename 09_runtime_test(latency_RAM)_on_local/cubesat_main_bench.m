%% ============================================================
%% CubeSat Battery + LSTM SOH test bench  (PC)  - latency, RAM, metrics
%% Derived from cubesat_main.m / cubesat_eps_main_fixed.m
%% ============================================================
clear; close all; clc;

%% ---------------- switches (read these) ----------------
USE_TRUE_LAG       = false;   % false = SOH_lag_1 is the model's OWN previous output (deployment-like)
                              % true  = SOH_lag_1 is the simulator's true SOH  (leaks the answer)
MATCH_TRAINING_ORBIT = true;  % true = per-cycle eclipse jitter + alternating T_amb_eclipse, like the
                              % training data (eclipse_flag mean 0.5, delta_T_ambient = +/-8).
                              % cubesat_main.m had neither, so those 2 features were constant.
MODEL_FILE         = 'cubesat_soh_lstm__int8.tflite';
SKIP_FIRST         = 30;      % cycles ignored in the metrics (window filling)

p = cubesat_params();
rng(42);

pyenv('Version', 'C:\Users\Amine\Desktop\files _matlab_test_bench\CubeSat_V4_Final\CubeSat_V4\tf_env\Scripts\python.exe', ...
      'ExecutionMode','OutOfProcess');
module = py.importlib.import_module('lstm_core_8_tflite');
py.importlib.reload(module);
py_predictor = module.CubeSatSOHPredict(pyargs('model_path', MODEL_FILE, 'pad_start', false));
fprintf('LSTM predictor loaded (true_lag=%d, training_orbit=%d)\n\n', USE_TRUE_LAG, MATCH_TRAINING_ORBIT);

tic_total_sim = tic;
n_max = 8000;
cycleNums = (1:n_max)';
SOH_true = zeros(n_max,1);
SOH_lstm = nan(n_max,1);

fields = {'V_mean_V','V_min_V','V_max_V','Tavg_C','Tmin_C','Tmax_C',...
          'QD_Ah','QC_Ah','IR_ohm','DoD','discharge_time_min','T_amb_K'};
history = struct();
for f = 1:numel(fields), history.(fields{f}) = zeros(n_max,1); end

x = [0.99; p.T_operating_min + 5.0; 1.0];

ordered_features = {'V_mean_V','V_spread','V_mean_rolling','V_mean_lag_1','V_min_fade', ...
    'Tavg_C','thermal_range','delta_T_ambient','cold_cycle_count','Tavg_rolling', ...
    'eclipse_flag','QD_Ah','coulombic_eff','discharge_C_rate','capacity_retention', ...
    'QD_rolling','cycle','cumul_Ah','SOH_lag_1','QD_diff'};
feature_log = nan(n_max, numel(ordered_features));
last_known_soh = 1.0;

for cycle = 1:n_max

    %% 1) orbit parameters for this cycle
    p_cycle = p;
    if MATCH_TRAINING_ORBIT
        eclipse_min  = max(32.0, min(38.0, 35.0 + 1.0*randn()));
        sunlight_min = max(53.0, min(57.0, 55.0 + 0.8*randn()));
        p_cycle.eclipse_time  = eclipse_min  * 60;
        p_cycle.sunlight_time = sunlight_min * 60;
        if mod(cycle,2) == 1
            p_cycle.T_amb_eclipse = p.T_operating_min;         % -10 C
        else
            p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;   %  -2 C
        end
    end

    %% 2) simulate + history
    row = simulate_cycle(x, p_cycle, cycle);
    SOH_true(cycle) = row.SOH_end;
    for f = 1:numel(fields), history.(fields{f})(cycle) = row.(fields{f}); end
    x = row.x_next;

    %% 3) features
    feat_struct = compute_features_fixed(cycle, row, history);
    if cycle == 1
        feat_struct.SOH_lag_1 = 1.0;
    elseif USE_TRUE_LAG
        feat_struct.SOH_lag_1 = SOH_true(cycle-1);
    else
        if ~isnan(SOH_lstm(cycle-1)), last_known_soh = SOH_lstm(cycle-1); end
        feat_struct.SOH_lag_1 = last_known_soh;
    end

    py_dict = py.dict();
    for i = 1:numel(ordered_features)
        key = ordered_features{i};
        val = feat_struct.(key);
        if isnan(val) || isinf(val), val = 0.0; end
        feature_log(cycle,i) = val;
        py_dict{key} = val;
    end

    %% 4) predict. Cycle 1 is NOT fed: in training the first row was dropped
    %%    (its lag features are undefined), so no window ever contained it.
    if cycle >= 2
        soh_pred = py_predictor.predict_soh(py_dict);
        if isequal(soh_pred, py.None)
            SOH_lstm(cycle) = NaN;
        else
            SOH_lstm(cycle) = double(soh_pred);
        end
    end

    if mod(cycle,500)==0 || cycle==1 || SOH_true(cycle)<0.75
        fprintf('Cycle %4d | True SOH: %.4f | Pred SOH: %.4f | QD: %.3f Ah\n', ...
            cycle, SOH_true(cycle), SOH_lstm(cycle), row.QD_Ah);
    end
    if SOH_true(cycle) < 0.70
        fprintf('\nEOL reached at cycle %d | SOH = %.4f\n', cycle, SOH_true(cycle));
        break;
    end
end

idx = 1:cycle;
total_sim_s = toc(tic_total_sim);
fprintf('\nSimulation finished (%d cycles) in %.2f s wall-clock (includes the MATLAB ODE solver)\n', cycle, total_sim_s);

%% ---------------- save results ----------------
T_result = table(cycleNums(idx), SOH_true(idx), SOH_lstm(idx), ...
    'VariableNames', {'cycle','SOH_true','SOH_lstm'});
writetable(T_result, 'CubeSat_REALTIME_LSTM_FINAL.csv');
T_feat = array2table(feature_log(idx,:), 'VariableNames', ordered_features);
writetable(T_feat, 'CubeSat_REALTIME_FEATURE_LOG.csv');

%% ---------------- ACCURACY METRICS (computed here, no Python needed) ----------------
ok  = idx' > SKIP_FIRST & ~isnan(SOH_lstm(idx));
t   = SOH_true(idx); t = t(ok);
y   = SOH_lstm(idx); y = y(ok);
e   = y - t;
fprintf('\n================ ACCURACY (cycles > %d, n=%d) ================\n', SKIP_FIRST, numel(t));
fprintf('MAE            : %.3f %% SOH\n', mean(abs(e))*100);
fprintf('RMSE           : %.3f %% SOH\n', sqrt(mean(e.^2))*100);
fprintf('Max abs error  : %.3f %% SOH\n', max(abs(e))*100);
fprintf('Bias (pred-true): %+.3f %% SOH (positive = optimistic)\n', mean(e)*100);
fprintf('MAPE           : %.3f %%\n', mean(abs(e)./abs(t))*100);
fprintf('R^2            : %.5f\n', 1 - sum(e.^2)/sum((t-mean(t)).^2));
fprintf('Final cycle    : pred %.4f vs true %.4f\n', y(end), t(end));
fprintf('=============================================================\n');

%% ---------------- LATENCY + RAM ----------------
py_predictor.print_performance_report();
perf = struct(py_predictor.get_performance_report());
if isfield(perf,'n_inferences') && double(perf.n_inferences) > 0
    hist     = py_predictor.get_latency_history();
    total_ms = double(hist{1});
    infer_ms = double(hist{2});
    ram_mb   = double(hist{3});
    n_inf    = numel(total_ms);
    infer_cycle_idx = (cycle - n_inf + 1):cycle;
    T_perf = table(infer_cycle_idx', total_ms', infer_ms', 'VariableNames', {'cycle','total_latency_ms','pure_inference_ms'});
    if ~isempty(ram_mb), T_perf.ram_mb = ram_mb'; end
    writetable(T_perf, 'CubeSat_LSTM_PERF.csv');

    figure('Position',[100 100 1100 600]);
    subplot(2,1,1);
    plot(infer_cycle_idx, total_ms, 'Color',[0.85 0.33 0.10]); hold on;
    plot(infer_cycle_idx, infer_ms, 'Color',[0.00 0.45 0.74]);
    yline(mean(total_ms(min(5,end):end)), 'k--');
    legend('Total (preproc+infer)','Pure invoke()','Mean total','Location','best');
    title('Inference latency per cycle (Python side only; first calls are slower)');
    xlabel('Cycle'); ylabel('ms'); grid on;
    subplot(2,1,2);
    if ~isempty(ram_mb)
        plot(infer_cycle_idx, ram_mb, 'Color',[0.47 0.67 0.19]);
        title('Python process RSS per cycle (includes TensorFlow itself, not the MCU RAM need)');
        xlabel('Cycle'); ylabel('MB'); grid on;
    else
        text(0.5,0.5,'psutil not installed: RAM not tracked','HorizontalAlignment','center'); axis off;
    end
end

%% ---------------- PLOT ----------------
figure('Position',[100 100 1200 700]);
plot(cycleNums(idx), SOH_true(idx), 'b-', 'LineWidth', 2.5); hold on;
valid = ~isnan(SOH_lstm(idx));
plot(cycleNums(valid), SOH_lstm(valid), 'r--', 'LineWidth', 2.0);
yline(0.70,'k--','EOL 70%'); legend('True SOH','Predicted SOH (raw)');
title(sprintf('LSTM SOH - true_lag=%d, training_orbit=%d', USE_TRUE_LAG, MATCH_TRAINING_ORBIT));
xlabel('Cycle'); ylabel('SOH'); grid on;
