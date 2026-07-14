# Stufe 2: Trajektorien-Inspektion + kalibrierter CUSUM-Detektor
# Frage: Wann ist der echte Kontakt, und wie frueh koennte ein
# baseline-korrigierter Detektor ihn finden — bei 0 False Positives im Freilauf?
import numpy as np
import pandas as pd

DIR = r"C:\Users\leonp\Documents\MultiGrasp\multigrasp\artifacts\analysis"
joints = ["servo6", "servo7", "servo8", "servo9"]

bl = pd.read_csv(DIR + r"\servo_analysis_precision_20260708_122810.csv")
bl_close = bl[bl["phase"] == "CLOSE"].copy()
bl_close["rstep"] = bl_close.groupby("cycle").cumcount()

ct = pd.read_csv(DIR + r"\contact_latency_precision_20260708_124132.csv")

# Baseline: mean-Trajektorie + std pro rstep, Startup-Steps (0-14) ausgenommen
START = 15
bl_stats = {}
for j in joints:
    piv = bl_close.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
    bl_stats[j] = pd.DataFrame({"m": piv.mean(axis=1), "s": piv.std(axis=1)})

# ------------------------------------------------ A. TRAJEKTORIEN (Zyklus 1)
print("=" * 100)
print("A. RESIDUUM-TRAJEKTORIEN Kugel-Test, Zyklus 1 (q_delta - baseline_mean), alle 8 Steps")
print("=" * 100)
cyc1 = ct[ct["cycle"] == 1].set_index("step")
hdr = f"{'step':>4} {'qt6':>6} " + " ".join(f"{'r'+j[-1]:>7}" for j in joints) + "   qm6    qm7    qm8    qm9"
print(hdr)
for k in range(START, 200, 8):
    vals = []
    for j in joints:
        r = cyc1.loc[k, f"{j}_q_delta"] - bl_stats[j].loc[k, "m"]
        vals.append(f"{r:+.4f}")
    qms = " ".join(f"{cyc1.loc[k, f'{j}_q_measured']:.3f}" for j in joints)
    print(f"{k:>4} {cyc1.loc[k, 'servo6_q_target']:>6.3f} " + " ".join(f"{v:>7}" for v in vals) + "   " + qms)

# ------------------------------------------ B. CUSUM-DETEKTOR, kalibriert
print()
print("=" * 100)
print("B. CUSUM auf Residuum. Alarm-Schwelle = kleinste Schwelle mit 0 False Positives")
print("   ueber alle 10 Freilauf-Zyklen (Leave-one-out waere besser, hier: gleiche Daten -> optimistisch)")
print("=" * 100)

DRIFT = 0.003  # Toleranz pro Step, ~1x Zyklus-zu-Zyklus-Std

def cusum(resid, drift=DRIFT):
    s = 0.0
    out = []
    for r in resid:
        s = max(0.0, s + r - drift)
        out.append(s)
    return np.array(out)

for j in joints:
    st = bl_stats[j]
    # False-Positive-Kalibrierung: CUSUM auf jedem Freilauf-Zyklus (Residuum vs. mean der ANDEREN 9)
    piv = bl_close.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
    max_free_cusum = 0.0
    for c in piv.columns:
        others = piv.drop(columns=c).mean(axis=1)
        resid = (piv[c] - others).iloc[START:].to_numpy()
        max_free_cusum = max(max_free_cusum, cusum(resid).max())
    alarm = max_free_cusum * 1.2  # 20% Marge
    print(f"\n{j}: max CUSUM im Freilauf={max_free_cusum:.4f} -> Alarmschwelle={alarm:.4f}")
    for cyc, grp in ct.groupby("cycle"):
        grp = grp.set_index("step")
        idx = grp.index.intersection(st.index)
        idx = idx[idx >= START]
        resid = (grp.loc[idx, f"{j}_q_delta"] - st.loc[idx, "m"]).to_numpy()
        cs = cusum(resid)
        hit = np.argmax(cs > alarm) if (cs > alarm).any() else None
        if hit is not None:
            k = idx[hit]
            print(f"  Zyklus {cyc}: CUSUM-Alarm @step {k} (q_target={grp.loc[k, f'{j}_q_target']:.3f})")
        else:
            print(f"  Zyklus {cyc}: kein Alarm")

# --------------------------- C. Vergleich: aktuelle Policy (Threshold 0.05)
print()
print("=" * 100)
print("C. AKTUELLE POLICY: erster Step mit q_delta > 0.05 (so detektiert das RL heute)")
print("=" * 100)
for j in joints:
    firsts = []
    for cyc, grp in ct.groupby("cycle"):
        grp = grp.set_index("step")
        over = grp[grp[f"{j}_q_delta"] > 0.05]
        firsts.append(over.index.min() if len(over) else None)
    print(f"{j}: erster Step ueber 0.05 pro Zyklus: {firsts}")

# Und im FREILAUF: wie oft ueberschreitet q_delta 0.05 (False Positives heute)?
print("\nFreilauf-Check Threshold 0.05 (haette das RL im Leerlauf 'Kontakt' gesehen?):")
for j in joints:
    n = 0
    for c, grp in bl_close.groupby("cycle"):
        if (grp[f"{j}_q_delta"].iloc[START:] > 0.05).any():
            n += 1
    print(f"  {j}: {n}/10 Freilauf-Zyklen mit q_delta > 0.05")

# ------------------------------ D. MCP-only Tests: wo liegt das max Residuum?
print()
print("=" * 100)
print("D. MCP-ONLY Tests: Residuum-Verlauf (jeder 10. Step), Kugel vs. Wuerfel, Zyklus 2")
print("=" * 100)
for name, path in [("Kugel", r"\contact_latency_precision_20260708_133935.csv"),
                   ("Wuerfel", r"\contact_latency_precision_20260708_134024.csv")]:
    d = pd.read_csv(DIR + path)
    c2 = d[d["cycle"] == 2].set_index("step")
    print(f"\n{name} (Zyklus 2):")
    line = []
    for k in range(START, min(200, c2.index.max() + 1), 10):
        r6 = c2.loc[k, "servo6_q_delta"] - c2.loc[k, "servo6_baseline_delta"]
        r8 = c2.loc[k, "servo8_q_delta"] - c2.loc[k, "servo8_baseline_delta"]
        line.append(f"s{k}:{r6:+.3f}/{r8:+.3f}")
    print("  " + "  ".join(line))

# ------------------------------ E. servo7/9: PIP-Cap-Effekt (Signal nach Cap)
print()
print("=" * 100)
print("E. PIP nach Cap (q_target=0.5, ab step ~100): q_delta Freilauf vs. Kugel")
print("   Freilauf: Servo holt auf -> q_delta klein. Kontakt: Objekt blockiert -> q_delta bleibt.")
print("=" * 100)
for j in ["servo7", "servo9"]:
    piv = bl_close.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
    late_free = piv.loc[120:199].to_numpy().ravel()
    print(f"\n{j} Freilauf steps 120-199: mean={np.nanmean(late_free):+.4f} "
          f"std={np.nanstd(late_free):.4f} max={np.nanmax(late_free):+.4f}")
    for cyc, grp in ct.groupby("cycle"):
        grp = grp.set_index("step")
        late = grp.loc[120:199, f"{j}_q_delta"]
        print(f"  Kugel Zyklus {cyc}: steps 120-199 mean={late.mean():+.4f} max={late.max():+.4f}")
