# Tiefenanalyse der AR10-Sensordaten:
# 1. Baseline-Reproduzierbarkeit (Freilauf, Zyklus-zu-Zyklus)
# 2. Kontaktsignal Kugel vs. Baseline (Level-Detektion)
# 3. Velocity-/Stall-Detektion (Steigung von q_measured)
# 4. Raw-ADC-Quantisierung
import numpy as np
import pandas as pd

DIR = r"C:\Users\leonp\Documents\MultiGrasp\multigrasp\artifacts\analysis"

pd.set_option("display.width", 200)
np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------- 1. BASELINE
print("=" * 78)
print("1. BASELINE-REPRODUZIERBARKEIT (servo_analysis 122810, Freilauf, 4 Joints)")
print("=" * 78)
bl = pd.read_csv(DIR + r"\servo_analysis_precision_20260708_122810.csv")
print("Phasen:", bl["phase"].unique(), " Zyklen:", sorted(bl["cycle"].unique()))
bl_close = bl[bl["phase"] == "CLOSE"].copy()
# step ist globaler Zaehler -> relativer Index innerhalb (cycle, phase)
bl_close["rstep"] = bl_close.groupby("cycle").cumcount()
print("CLOSE-Steps pro Zyklus:", bl_close.groupby("cycle")["step"].count().tolist())

joints = ["servo6", "servo7", "servo8", "servo9"]

# Pro Joint: Trajektorien-Matrix (Zyklen x Steps), dann Zyklus-zu-Zyklus-Std
for j in joints:
    piv = bl_close.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
    piv = piv.dropna()
    mean_traj = piv.mean(axis=1)
    std_traj = piv.std(axis=1)   # Streuung ZWISCHEN Zyklen am selben Step
    print(f"\n{j}:")
    print(f"  q_delta gesamt:      mean={piv.values.mean():+.4f}  "
          f"min={piv.values.min():+.4f}  max={piv.values.max():+.4f}")
    print(f"  Zyklus-zu-Zyklus-Std am selben Step: "
          f"median={std_traj.median():.4f}  max={std_traj.max():.4f}")
    print(f"  -> Baseline-korrigierter 3-sigma-Threshold (median): "
          f"{3*std_traj.median():.4f}   (max: {3*std_traj.max():.4f})")
    # Wie sieht die mittlere Trajektorie aus (alle 20 Steps)?
    sel = mean_traj.iloc[::20]
    print("  mean q_delta ueber Steps:",
          "  ".join(f"s{int(s)}:{v:+.3f}" for s, v in sel.items()))

# ------------------------------------------------- 2. KONTAKT KUGEL (4 Joints)
print()
print("=" * 78)
print("2. KONTAKTSIGNAL KUGEL (contact_latency 124132) vs. Baseline")
print("=" * 78)
ct = pd.read_csv(DIR + r"\contact_latency_precision_20260708_124132.csv")
print("Zyklen:", sorted(ct["cycle"].unique()),
      " Steps/Zyklus:", ct.groupby("cycle")["step"].count().tolist())

# Baseline-Statistik pro Step aus dem Freilauf (mean + std ueber Zyklen)
bl_stats = {}
for j in joints:
    piv = bl_close.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta").dropna()
    bl_stats[j] = pd.DataFrame({"bl_mean": piv.mean(axis=1), "bl_std": piv.std(axis=1)})

for j in joints:
    print(f"\n--- {j} ---")
    st = bl_stats[j]
    for cyc, grp in ct.groupby("cycle"):
        grp = grp.set_index("step")
        idx = grp.index.intersection(st.index)
        resid = grp.loc[idx, f"{j}_q_delta"] - st.loc[idx, "bl_mean"]
        z = resid / st.loc[idx, "bl_std"].clip(lower=1e-4)
        # Erster Step mit z>3 fuer >=3 aufeinanderfolgende Steps (robust)
        hits = (z > 3).astype(int)
        run = hits.rolling(3).sum()
        first = run[run >= 3].index.min()
        peak_step = resid.idxmax()
        print(f"  Zyklus {cyc}: max Residuum={resid.max():+.4f} @step {peak_step} "
              f"(q_target={grp.loc[peak_step, f'{j}_q_target']:.3f}), "
              f"max z={z.max():5.1f}, "
              f"erster stabiler z>3: step {first if pd.notna(first) else '—'}"
              + (f" (q_target={grp.loc[first, f'{j}_q_target']:.3f})" if pd.notna(first) else ""))

# ---------------------------------------------- 3. VELOCITY-/STALL-DETEKTION
print()
print("=" * 78)
print("3. STALL-DETEKTION: Steigung von q_measured (Fenster=8 Steps)")
print("   Freilauf: Steigung ~ delta_norm (0.005/Step). Kontakt: Steigung -> 0")
print("=" * 78)
W = 8

def slope(series, w=W):
    # Rollierende Steigung per linearer Regression ueber w Steps
    x = np.arange(w)
    x = x - x.mean()
    denom = (x ** 2).sum()
    return series.rolling(w).apply(lambda v: float(np.dot(x, v)) / denom, raw=True)

for j in joints:
    print(f"\n--- {j} ---")
    # Freilauf-Referenz: Steigungsverteilung waehrend CLOSE (nur Rampe, q_target<0.95)
    sl_free_all = []
    for cyc, grp in bl_close.groupby("cycle"):
        grp = grp.sort_values("step")
        ramp = grp[grp[f"{j}_q_target"] < 0.95]
        sl = slope(ramp[f"{j}_q_measured"]).dropna()
        sl_free_all.append(sl)
    sl_free = pd.concat(sl_free_all)
    p5 = sl_free.quantile(0.05)
    print(f"  Freilauf-Steigung: mean={sl_free.mean():.5f}  std={sl_free.std():.5f}  "
          f"p5={p5:.5f}  min={sl_free.min():.5f}")

    # Kontakt-Zyklen: erster Step, an dem Steigung < p5/2 (Stall) waehrend Rampe
    for cyc, grp in ct.groupby("cycle"):
        grp = grp.sort_values("step").set_index("step")
        ramp = grp[grp[f"{j}_q_target"] < 0.95]
        sl = slope(ramp[f"{j}_q_measured"]).dropna()
        stall_thr = max(0.0005, p5 * 0.5)
        stalled = sl[sl < stall_thr]
        first = stalled.index.min()
        sl_min = sl.min()
        print(f"  Zyklus {cyc}: min Steigung={sl_min:.5f}, "
              f"erster Stall (<{stall_thr:.5f}): step "
              f"{first if pd.notna(first) else '—'}"
              + (f" (q_target={grp.loc[first, f'{j}_q_target']:.3f}, "
                 f"q_meas={grp.loc[first, f'{j}_q_measured']:.3f})" if pd.notna(first) else ""))

# Wie oft wuerde der Stall-Detektor im FREILAUF falsch ausloesen?
print("\n  False-Positive-Check Freilauf (Stall-Kriterium auf Baseline angewandt):")
for j in joints:
    fp = 0
    tot = 0
    for cyc, grp in bl_close.groupby("cycle"):
        grp = grp.sort_values("step")
        ramp = grp[grp[f"{j}_q_target"] < 0.95]
        sl = slope(ramp[f"{j}_q_measured"]).dropna()
        # gleiche Schwelle wie oben
        sl_free = sl  # hier ist alles Freilauf
        fp += int((sl < 0.0005).sum() > 0)
        tot += 1
    print(f"    {j}: {fp}/{tot} Freilauf-Zyklen mit mind. 1 Stall-Fehlalarm (Schwelle 0.0005)")

# ---------------------------------------------------------------- 4. RAW ADC
print()
print("=" * 78)
print("4. RAW-ADC-QUANTISIERUNG (raw_adc 133827, servo6/8)")
print("=" * 78)
adc = pd.read_csv(DIR + r"\raw_adc_precision_20260708_133827.csv")
for j in ["servo6", "servo8"]:
    r = adc[f"{j}_raw_adc"]
    diffs = r.diff().dropna()
    print(f"{j}: ADC-Range [{r.min()}, {r.max()}], "
          f"Schrittweite pro Step: mean={diffs.mean():.2f}, "
          f"unique diffs: {sorted(diffs.unique())[:12]}")
    span = r.max() - r.min()
    print(f"   -> 1 ADC-Count = {1.0/span:.5f} q-Einheiten; "
          f"delta_norm 0.005 = {0.005*span:.1f} Counts/Step")

# --------------------------------------- 5. MCP-ONLY TESTS (Kugel vs Wuerfel)
print()
print("=" * 78)
print("5. MCP-ONLY (133935 Kugel / 134024 Wuerfel): Residuum vs. Baseline")
print("=" * 78)
for name, path in [("Kugel", r"\contact_latency_precision_20260708_133935.csv"),
                   ("Wuerfel", r"\contact_latency_precision_20260708_134024.csv")]:
    d = pd.read_csv(DIR + path)
    print(f"\n{name}:")
    for j in ["servo6", "servo8"]:
        resid = d[f"{j}_q_delta"] - d[f"{j}_baseline_delta"]
        print(f"  {j}: Residuum mean={resid.mean():+.4f}  std={resid.std():.4f}  "
              f"max={resid.max():+.4f}")
