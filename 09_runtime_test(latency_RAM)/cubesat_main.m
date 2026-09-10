%% ============================================================
%% CubeSat Battery + REAL-TIME LSTM SOH Prediction (FINAL FIXED)
%% ============================================================

clear; close all; clc;

p = cubesat_params();

fprintf('=== CubeSat REAL-TIME LSTM SOH Prediction (FINAL FIXED) ===\n\n');

%% ============================================================
%% PYTHON INIT
%% ============================================================
pyenv('Version', 'C:\Users\Amine\Desktop\files _6\CubeSat_V4_Final\CubeSat_V4\tf_env\Scripts\python.exe', ...
      'ExecutionMode','OutOfProcess');

module = py.importlib.import_module('lstm_core_7_tflite');
py.importlib.reload(module);
py_predictor = module.CubeSatSOHPredict();

fprintf('✅ LSTM Predictor Loaded\n');

%% ============================================================
%% TOTAL WALL-CLOCK TIMER FOR THE WHOLE RUN
%% ============================================================
tic_total_sim = tic;

%% ============================================================
%% CONFIG
%% ============================================================
n_max = 8000;
cycleNums = (1:n_max)';

SOH_true = zeros(n_max,1);
SOH_lstm = nan(n_max,1);

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
    
    %% 1️⃣ Simulate cycle
    row = simulate_cycle(x, p, cycle);
    SOH_true(cycle) = row.SOH_end;
    
    %% 2️⃣ Update history
    for f = 1:numel(fields)
        history.(fields{f})(cycle) = row.(fields{f});
    end
    
    x = row.x_next;
    
    %% 3️⃣ Feature engineering
    feat_struct = compute_features(cycle, row, history);
    
    %% 🔥 TRUE SOH lag
    if cycle == 1
     feat_struct.SOH_lag_1 = 1.0;
   else
     feat_struct.SOH_lag_1 = SOH_true(cycle-1);
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
    if mod(cycle,500)==0 || cycle==1 || SOH_true(cycle)<0.75
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

total_sim_s = toc(tic_total_sim);
fprintf('\n✅ Simulation finished (%d cycles) in %.2f s wall-clock\n', cycle, total_sim_s);

%% ============================================================
%% INFERENCE LATENCY + RAM REPORT (pulled from the Python object)
%% ============================================================
py_predictor.print_performance_report();

perf = py_predictor.get_performance_report();
perf = struct(perf);   % py.dict -> MATLAB struct

if isfield(perf, 'n_inferences') && double(perf.n_inferences) > 0

    % --- Pull per-cycle latency/RAM history for plotting ---
    hist = py_predictor.get_latency_history();
    total_ms = double(hist{1});   % preprocessing + inference, per call
    infer_ms = double(hist{2});   % pure interpreter.invoke(), per call
    ram_mb   = double(hist{3});   % process RSS, per call (empty if psutil missing)

    n_inf = numel(total_ms);
    infer_cycle_idx = (cycle - n_inf + 1):cycle;   % cycles where inference actually ran

    T_perf = table(infer_cycle_idx', total_ms', infer_ms', ...
        'VariableNames', {'cycle','total_latency_ms','pure_inference_ms'});
    if ~isempty(ram_mb)
        T_perf.ram_mb = ram_mb';
    end
    writetable(T_perf, 'CubeSat_LSTM_PERF.csv');

    fprintf('📄 Per-cycle latency/RAM log saved to CubeSat_LSTM_PERF.csv\n');

    %% -------------------------------------------------------
    %% PLOT: latency per cycle
    %% -------------------------------------------------------
    figure('Position',[100 100 1100 600]);

    subplot(2,1,1);
    plot(infer_cycle_idx, total_ms, 'Color',[0.85 0.33 0.10], 'LineWidth', 1.2); hold on;
    plot(infer_cycle_idx, infer_ms, 'Color',[0.00 0.45 0.74], 'LineWidth', 1.2);
    yline(mean(total_ms), 'k--');
    legend('Total (preproc+infer)','Pure TFLite invoke()','Mean total','Location','best');
    title('Inference Latency per Cycle');
    xlabel('Cycle'); ylabel('Latency (ms)'); grid on;

    subplot(2,1,2);
    if ~isempty(ram_mb)
        plot(infer_cycle_idx, ram_mb, 'Color',[0.47 0.67 0.19], 'LineWidth', 1.2);
        title('Python Process RAM (RSS) per Cycle');
        xlabel('Cycle'); ylabel('RAM (MB)'); grid on;
    else
        text(0.5, 0.5, 'psutil not installed: RAM not tracked', ...
            'HorizontalAlignment','center');
        axis off;
    end
else
    fprintf('⚠️  No performance data collected (0 inferences ran).\n');
end

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