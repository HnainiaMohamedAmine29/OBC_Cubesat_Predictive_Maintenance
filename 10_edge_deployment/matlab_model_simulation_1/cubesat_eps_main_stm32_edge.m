%% ============================================================
%% CubeSat Battery + STM32 SOH prediction - EDGE PREPROCESSING
%% MATLAB only simulates the battery and sends the RAW cycle data.
%% The STM32 computes the features, scales them, keeps the 30-cycle
%% window and runs the LSTM. (Derived from cubesat_eps_main_stm32_usb.m)
%% ============================================================
clear; close all; clc;

%% ---------------- settings ----------------
COM_PORT   = "COM6";      % ST-Link virtual COM port
BAUD_RATE  = 115200;      % must match huart1.Init.BaudRate
USE_TRUE_LAG = false;     % false = the BOARD uses its own previous prediction as SOH_lag_1 (no ground truth)
                          % true  = MATLAB sends the true SOH as SOH_lag_1 (only for comparing with the old run)
SKIP_FIRST = 30;          % metrics are printed for ALL cycles and for cycles > SKIP_FIRST (the seeded warm-up window is still filling)

p = cubesat_params();
rng(42);                  % same random orbit sequence as the old run -> directly comparable

ports = serialportlist("available");
if strlength(COM_PORT) == 0
    disp(ports'); error('Set COM_PORT.');
end
sp = serialport(COM_PORT, BAUD_RATE);
sp.Timeout = 10;
flush(sp);
fprintf('Connected to STM32 (%s @ %d baud) - edge preprocessing, true_lag=%d\n\n', COM_PORT, BAUD_RATE, USE_TRUE_LAG);

n_max = 8000;
SOH_true = zeros(n_max,1);
SOH_lstm = nan(n_max,1);
t_pre_us = nan(n_max,1);  t_inf_us = nan(n_max,1);  t_rt_ms = nan(n_max,1);
status_log = zeros(n_max,1,'uint8');
arena_used = NaN;

x = [0.99; p.T_operating_min + 5.0; 1.0];

for cycle = 1:n_max

    %% orbit parameters (identical to the old script / cubesat_run.m)
    eclipse_min  = max(32.0, min(38.0, 35.0 + 1.0 * randn()));
    sunlight_min = max(53.0, min(57.0, 55.0 + 0.8 * randn()));
    p_cycle = p;
    p_cycle.eclipse_time  = eclipse_min  * 60;
    p_cycle.sunlight_time = sunlight_min * 60;
    if mod(cycle, 2) == 1
        p_cycle.T_amb_eclipse = p.T_operating_min;
    else
        p_cycle.T_amb_eclipse = p.T_operating_min + 8.0;
    end

    %% simulate one cycle
    row = simulate_cycle(x, p_cycle, cycle);
    SOH_true(cycle) = row.SOH_end;
    x = row.x_next;

    %% send ONLY the raw row; the board does everything else
    if cycle == 1, lag_ext = 1.0; else, lag_ext = SOH_true(cycle-1); end
    tic_rt = tic;
    r = uart_soh_predict_raw(sp, cycle, row, lag_ext, USE_TRUE_LAG);
    t_rt_ms(cycle) = toc(tic_rt) * 1e3;

    SOH_lstm(cycle)   = r.soh;
    t_pre_us(cycle)   = r.t_pre_us;
    t_inf_us(cycle)   = r.t_inf_us;
    status_log(cycle) = r.status;
    arena_used        = r.arena;

    if mod(cycle,200)==0 || cycle==1 || SOH_true(cycle)<0.75
        fprintf('Cycle %4d | True SOH: %.4f | Pred SOH: %.4f | board: preproc %5.0f us, infer %6.0f us\n', ...
            cycle, SOH_true(cycle), SOH_lstm(cycle), t_pre_us(cycle), t_inf_us(cycle));
    end
    if SOH_true(cycle) < 0.70
        fprintf('\nEOL reached at cycle %d | SOH = %.4f\n', cycle, SOH_true(cycle));
        break;
    end
end
clear sp;

%% ---------------- save ----------------
idx = 1:cycle;
T = table(idx', SOH_true(idx), SOH_lstm(idx), t_pre_us(idx), t_inf_us(idx), t_rt_ms(idx), status_log(idx), ...
    'VariableNames', {'cycle','SOH_true','SOH_lstm','t_preproc_us','t_infer_us','roundtrip_ms','status'});
writetable(T, 'CubeSat_STM32_EDGE_FINAL.csv');
% the same file works with:  python compute_metrics.py CubeSat_STM32_EDGE_FINAL.csv --skip 30

%% ---------------- ACCURACY ----------------
for skip = [0 SKIP_FIRST]
    ok = idx' > skip & ~isnan(SOH_lstm(idx));
    t = SOH_true(idx); t = t(ok);  y = SOH_lstm(idx); y = y(ok);  e = y - t;
    fprintf('\n================ ACCURACY (cycles > %d, n=%d) ================\n', skip, numel(t));
    fprintf('MAE             : %.3f %% SOH\n', mean(abs(e))*100);
    fprintf('RMSE            : %.3f %% SOH\n', sqrt(mean(e.^2))*100);
    fprintf('Max abs error   : %.3f %% SOH\n', max(abs(e))*100);
    fprintf('Bias (pred-true): %+.3f %% SOH (positive = optimistic)\n', mean(e)*100);
    fprintf('R^2             : %.5f\n', 1 - sum(e.^2)/sum((t-mean(t)).^2));
    fprintf('Final cycle     : pred %.4f vs true %.4f\n', y(end), t(end));
end

%% ---------------- LATENCY / MEMORY (measured ON THE BOARD, DWT cycle counter @ SystemCoreClock) ----------------
pre = t_pre_us(idx); inf_ = t_inf_us(idx); rt = t_rt_ms(idx);
pr = @(v) sprintf('mean %.1f | p50 %.1f | p95 %.1f | max %.1f', mean(v,'omitnan'), prctile(v,50), prctile(v,95), max(v));
fprintf('\n================ BOARD LATENCY / MEMORY ================\n');
fprintf('preprocessing (features+scale+window): %s us\n', pr(pre));
fprintf('model inference (tflm_infer)          : %s us\n', pr(inf_));
fprintf('total compute on board                : mean %.2f ms\n', mean(pre + inf_, 'omitnan')/1000);
fprintf('MATLAB<->board round trip (UART+sim)  : %s ms\n', pr(rt));
fprintf('TFLM tensor arena actually used       : %.1f KB\n', arena_used/1024);
fprintf('status codes: OK=%d  DUPLICATE=%d  GAP=%d  INFER_ERR=%d  WARMUP=%d (only if the optional gate is on)\n', ...
    sum(status_log(idx)==0), sum(status_log(idx)==1), sum(status_log(idx)==2), sum(status_log(idx)==3), sum(status_log(idx)==4));

%% ---------------- plot ----------------
figure('Position',[100 100 1200 700]);
plot(idx, SOH_true(idx), 'b-', 'LineWidth', 2.5); hold on;
v = ~isnan(SOH_lstm(idx));
plot(idx(v), SOH_lstm(idx(v)), 'r--', 'LineWidth', 2.0);
yline(0.70,'k--','EOL 70%'); grid on;
legend('True SOH','Predicted SOH (STM32, edge preprocessing)');
title(sprintf('STM32 edge pipeline - true_lag=%d', USE_TRUE_LAG)); xlabel('Cycle'); ylabel('SOH');
