"""
Accuracy metrics from the MATLAB test-bench CSV (cycle, SOH_true, SOH_lstm).

python compute_metrics.py CubeSat_REALTIME_LSTM_FINAL.csv [--skip 30] [--png report.png]
"""
import argparse, numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument('csv'); p.add_argument('--skip', type=int, default=30, help='ignore first N cycles (window filling)')
p.add_argument('--png', default=None); p.add_argument('--eol', type=float, default=0.70)
a = p.parse_args()

d = pd.read_csv(a.csv)
d = d[d.cycle > a.skip].dropna(subset=['SOH_lstm']).reset_index(drop=True)
t, y = d.SOH_true.values.astype(float), d.SOH_lstm.values.astype(float)
e = y - t                                              # signed error (pred - true), 1-D on purpose
ae = np.abs(e)
ss_res, ss_tot = np.sum(e ** 2), np.sum((t - t.mean()) ** 2)

print(f'rows evaluated        : {len(d)}  (cycles {d.cycle.min()}-{d.cycle.max()}, first {a.skip} skipped)')
print(f'MAE                   : {ae.mean()*100:.3f} % SOH')
print(f'RMSE                  : {np.sqrt((e**2).mean())*100:.3f} % SOH')
print(f'max abs error         : {ae.max()*100:.3f} % SOH  (cycle {int(d.cycle[ae.argmax()])})')
print(f'median / p95 abs error: {np.median(ae)*100:.3f} / {np.percentile(ae,95)*100:.3f} % SOH')
print(f'bias (pred - true)    : {e.mean()*100:+.3f} % SOH   (positive = optimistic)')
print(f'MAPE                  : {np.mean(ae/np.maximum(np.abs(t),1e-9))*100:.3f} %')
print(f'R2                    : {1 - ss_res/ss_tot:.5f}')
print(f'final-cycle error     : pred {y[-1]:.4f} vs true {t[-1]:.4f}  ({(y[-1]-t[-1])*100:+.2f} % SOH)')

print('\nerror by true-SOH band (MAE % / bias %):')
for lo, hi in [(0.97, 1.01), (0.90, 0.97), (0.80, 0.90), (0.75, 0.80), (0.0, 0.75)]:
    m = (t >= lo) & (t < hi)
    if m.sum(): print(f'  SOH {lo:.2f}-{min(hi,1.0):.2f}: n={m.sum():5d}  MAE {ae[m].mean()*100:6.3f}  bias {e[m].mean()*100:+6.3f}')

true_eol = d.cycle[t < a.eol]; pred_eol = d.cycle[y < a.eol]
print(f'\nEOL ({a.eol:.2f}): true crossing at cycle {int(true_eol.iloc[0]) if len(true_eol) else "not reached (run stops at EOL)"}; '
      f'last true cycle {int(d.cycle.iloc[-1])}; ' + (f'prediction crosses at cycle {int(pred_eol.iloc[0])}' if len(pred_eol) else 'prediction NEVER crosses EOL'))
flat = np.mean(np.abs(np.diff(y)) < 1e-9) * 100
print(f'flat steps in prediction: {flat:.1f} % of cycles (high value = staircase / filtering)')

if a.png:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(d.cycle, t, label='true'); ax[0].plot(d.cycle, y, '--', label='LSTM'); ax[0].axhline(a.eol, color='k', ls=':'); ax[0].legend(); ax[0].set_ylabel('SOH'); ax[0].grid(alpha=.3)
    ax[1].plot(d.cycle, e * 100); ax[1].axhline(0, color='k', lw=.5); ax[1].set_ylabel('error (% SOH)'); ax[1].set_xlabel('cycle'); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(a.png, dpi=150); print('saved', a.png)
