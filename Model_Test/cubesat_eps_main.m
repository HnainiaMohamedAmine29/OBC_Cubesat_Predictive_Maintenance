function cubesat_eps_main(N_max_in)

clear; clc; close all

%% ── PARAMÈTRES ─────────────────────────────────────────────
if nargin < 1, N_max_in = 7000; end

N_MAX         = N_max_in;
MIN_CYCLES    = 100;

rng(42);
p = cubesat_params();
p.SOH_eol = 0.70;

%% ── PYTHON INIT (IMPORTANT) ────────────────────────────────
pyenv("Version","C:\Users\Amine\Desktop\files_4\tf_env\Scripts\python.exe");
py.importlib.import_module("lstm_core");

%% ── PRÉALLOCATION ───────────────────────────────────────────
cycles_log   = zeros(N_MAX,1);
soh_true_log = zeros(N_MAX,1);
soh_pred_log = nan(N_MAX,1);

%% ── ÉTAT INITIAL ────────────────────────────────────────────
x = [0.99; p.T_operating_min + 5.0; 1.0];

fprintf('\n=== CubeSat EPS + LSTM REAL-TIME (NO SERVER) ===\n');
fprintf('Cycle | SOH_true | SOH_pred | SOC | Tavg | IR\n');
fprintf('------------------------------------------------\n');

actual_n = 0;

%% ── LOOP PRINCIPALE ────────────────────────────────────────
for k = 1:N_MAX

    %% 1. Simulation batterie
    row = simulate_cycle(x, p, k);

    actual_n = k;

    cycles_log(k)   = k;
    soh_true_log(k) = row.SOH_end;

    %% 2. UPDATE STATE
    x = row.x_next;

    %% 3. REAL TIME LSTM (après MIN_CYCLES)
    if k >= MIN_CYCLES

        try
            data = py.dict(pyargs( ...
                'cycle', k, ...
                'V_mean_V', row.V_mean_V, ...
                'V_spread', row.V_max_V - row.V_min_V, ...
                'V_mean_rolling', row.V_mean_V, ...
                'V_mean_lag_1', row.V_mean_V, ...
                'V_min_fade', row.V_min_V, ...
                'Tavg_C', row.Tavg_C, ...
                'thermal_range', row.Tmax_C - row.Tmin_C, ...
                'delta_T_ambient', row.Tavg_C - (p.T_operating_min), ...
                'cold_cycle_count', 0, ...
                'Tavg_rolling', row.Tavg_C, ...
                'eclipse_flag', 0, ...
                'QD_Ah', row.QD_Ah, ...
                'coulombic_eff', row.QD_Ah / max(row.QC_Ah,1e-6), ...
                'discharge_C_rate', row.QD_Ah / max(row.discharge_time_min/60,1e-6), ...
                'capacity_retention', row.QD_Ah / max(row.QD_Ah,1e-6), ...
                'QD_rolling', row.QD_Ah, ...
                'cumul_Ah', row.QD_Ah * k, ...
                'SOH_lag_1', row.SOH_end, ...
                'QD_diff', 0 ...
            ));

            soh_pred = py.lstm_core.predict(data);
            soh_pred_log(k) = double(soh_pred);

        catch ME
            warning("LSTM error cycle %d: %s", k, ME.message);
        end
    end

    %% 4. DISPLAY
    if mod(k,50)==0 || k<=5
        pred_str = " --- ";
        if ~isnan(soh_pred_log(k))
            pred_str = sprintf(" %.4f ", soh_pred_log(k));
        end

        fprintf('%5d | %.4f |%s| %.4f | %.2f | %.4f\n', ...
            k, row.SOH_end, pred_str, row.SOC_end, row.Tavg_C, row.IR_ohm);
    end

    %% 5. STOP CONDITION
    if row.SOH_end < p.SOH_eol
        fprintf("\nBATTERIE MORTE cycle %d\n", k);
        break;
    end
end

n = actual_n;

%% ── PLOT SIMPLE ─────────────────────────────────────────────
figure;
plot(soh_true_log,'b','LineWidth',1.5); hold on;
plot(soh_pred_log,'r--','LineWidth',1.5);
legend("SOH vrai","SOH LSTM");
xlabel("Cycle"); ylabel("SOH");
title("Real-Time Co-Simulation MATLAB + Python (NO SERVER)");
grid on;

end