import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# -------------------------------------------------------------------------
# 1. Recherche automatique du fichier CSV le plus récent
# -------------------------------------------------------------------------
csv_files = glob.glob('battery_dataset_fault.csv')
if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV 'battery_dataset_Panasonic_NCR18650B_2S1P*.csv' trouvé.")

latest_file = max(csv_files, key=os.path.getmtime)
print(f"Fichier chargé : {latest_file}")

# -------------------------------------------------------------------------
# 2. Lecture du CSV (gestion automatique des virgules décimales)
# -------------------------------------------------------------------------
try:
    df = pd.read_csv(latest_file)
    numeric_cols = ['SOC_start', 'SOC_end', 'DoD', 'IR_ohm', 'QD_Ah', 'QC_Ah',
                    'V_mean_V', 'V_min_V', 'V_max_V', 'Tavg_C', 'Tmin_C', 'Tmax_C',
                    'chargetime_min', 'discharge_time_min', 'SOH', 'T_amb_K']
    # Si les colonnes numériques sont lues comme des objets, on réessaye avec decimal=','
    if df[numeric_cols].dtypes.apply(lambda x: x == 'object').any():
        df = pd.read_csv(latest_file, decimal=',')
except Exception as e:
    print("Erreur lecture CSV, tentative avec séparateur virgule...")
    df = pd.read_csv(latest_file, decimal=',')

# -------------------------------------------------------------------------
# 3. Préparation des données
# -------------------------------------------------------------------------
cycles = df['cycle']
soh = df['SOH']
soc_start = df['SOC_start']
soc_end = df['SOC_end']
dod = df['DoD']
ir_mohm = df['IR_ohm'] * 1000          # conversion en mΩ
tavg = df['Tavg_C']
qd = df['QD_Ah']
qc = df['QC_Ah']
vmean = df['V_mean_V']

# -------------------------------------------------------------------------
# 4. Fonction pour créer une figure individuelle de qualité
# -------------------------------------------------------------------------
def plot_single(x, y, ylabel, title, filename, color='b', ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color=color, linewidth=2)
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure sauvegardée : {filename}")

# -------------------------------------------------------------------------
# 5. Génération des figures individuelles
# -------------------------------------------------------------------------

# 5.1 SOH
plot_single(cycles, soh, 'SOH', 'Évolution de l\'état de santé (SOH)',
            'SOH_vs_cycles.png', color='blue', ylim=(0.65, 1.02))

# 5.2 SOC_start, SOC_end et DoD (superposés sur un même graphique pour contexte)
plt.figure(figsize=(10, 6))
plt.plot(cycles, soc_start, 'g-', linewidth=2, label='SOC$_{start}$')
plt.plot(cycles, soc_end, 'r-', linewidth=2, label='SOC$_{end}$')
plt.plot(cycles, dod, 'k--', linewidth=2, label='DoD')
plt.xlabel('Cycle', fontsize=12)
plt.ylabel('SOC / DoD', fontsize=12)
plt.title('Évolution du SOC et de la profondeur de décharge', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('SOC_DoD_vs_cycles.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : SOC_DoD_vs_cycles.png")

# 5.3 Résistance interne
plot_single(cycles, ir_mohm, 'Résistance interne (mΩ)',
            'Évolution de la résistance interne moyenne',
            'IR_vs_cycles.png', color='magenta')

# 5.4 Température moyenne
plot_single(cycles, tavg, 'Température moyenne (°C)',
            'Évolution de la température de fonctionnement',
            'Tavg_vs_cycles.png', color='red')

# 5.5 Capacités QD et QC (superposés)
plt.figure(figsize=(10, 6))
plt.plot(cycles, qd, 'b-', linewidth=2, label='QD (capacité de décharge)')
plt.plot(cycles, qc, 'r-', linewidth=2, label='QC (capacité de charge)')
plt.xlabel('Cycle', fontsize=12)
plt.ylabel('Capacité (Ah)', fontsize=12)
plt.title('Évolution des capacités disponible et chargée', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Capacities_vs_cycles.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : Capacities_vs_cycles.png")

# 5.6 Tension moyenne du pack
plot_single(cycles, vmean, 'Tension moyenne (V)',
            'Évolution de la tension moyenne du pack',
            'Vmean_vs_cycles.png', color='black')

# -------------------------------------------------------------------------
# 6. (Optionnel) Tableau de bord combiné
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'CubeSat LEO – Vue d\'ensemble\n{os.path.basename(latest_file)}',
             fontsize=16, fontweight='bold')

# SOH
axes[0,0].plot(cycles, soh, 'b-', linewidth=1.5)
axes[0,0].set_xlabel('Cycle'); axes[0,0].set_ylabel('SOH')
axes[0,0].set_title('État de santé'); axes[0,0].grid(True, linestyle='--', alpha=0.6)
axes[0,0].set_ylim(0.65, 1.02)

# SOC & DoD
axes[0,1].plot(cycles, soc_start, 'g-', label='SOC$_{start}$')
axes[0,1].plot(cycles, soc_end, 'r-', label='SOC$_{end}$')
axes[0,1].plot(cycles, dod, 'k--', label='DoD')
axes[0,1].set_xlabel('Cycle'); axes[0,1].set_ylabel('SOC / DoD')
axes[0,1].set_title('SOC et DoD'); axes[0,1].legend(loc='best', fontsize=8)
axes[0,1].grid(True, linestyle='--', alpha=0.6)

# IR
axes[0,2].plot(cycles, ir_mohm, 'm-', linewidth=1.5)
axes[0,2].set_xlabel('Cycle'); axes[0,2].set_ylabel('IR (mΩ)')
axes[0,2].set_title('Résistance interne'); axes[0,2].grid(True, linestyle='--', alpha=0.6)

# Tavg
axes[1,0].plot(cycles, tavg, 'r-', linewidth=1.5)
axes[1,0].set_xlabel('Cycle'); axes[1,0].set_ylabel('Température (°C)')
axes[1,0].set_title('Température moyenne'); axes[1,0].grid(True, linestyle='--', alpha=0.6)

# Capacités
axes[1,1].plot(cycles, qd, 'b-', label='QD')
axes[1,1].plot(cycles, qc, 'r-', label='QC')
axes[1,1].set_xlabel('Cycle'); axes[1,1].set_ylabel('Capacité (Ah)')
axes[1,1].set_title('Capacités'); axes[1,1].legend(loc='best', fontsize=8)
axes[1,1].grid(True, linestyle='--', alpha=0.6)

# Vmean
axes[1,2].plot(cycles, vmean, 'k-', linewidth=1.5)
axes[1,2].set_xlabel('Cycle'); axes[1,2].set_ylabel('Tension (V)')
axes[1,2].set_title('Tension moyenne'); axes[1,2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('dashboard_combined.png', dpi=200, bbox_inches='tight')
plt.show()
print("Tableau de bord sauvegardé : dashboard_combined.png")

print("\n✅ Toutes les figures ont été générées avec succès.")