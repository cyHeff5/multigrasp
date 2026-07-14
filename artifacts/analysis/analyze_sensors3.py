# Stufe 3: Settle-Verhalten (Basis fuer "Micro-Pause"-Detektion) + Slow-Close
import numpy as np
import pandas as pd

DIR = r"C:\Users\leonp\Documents\MultiGrasp\multigrasp\artifacts\analysis"

# ------------------- A. Statische Positionen: wie leise ist der Sensor in Ruhe?
print("=" * 90)
print("A. POSITION-NOISE (statisch, 100 Samples/Position) — volle Tabelle")
print("=" * 90)
pn = pd.read_csv(DIR + r"\position_noise_summary_precision_20260708_133816.csv")
print(pn.to_string(index=False))

# ------------------- B. Settle-Verhalten: CLOSE -> HOLD_CLOSED Uebergang
print()
print("=" * 90)
print("B. SETTLE: q_delta in den ersten 15 Steps von HOLD_CLOSED (Freilauf, alle Zyklen)")
print("   -> Wie schnell kollabiert q_delta, wenn die Rampe stoppt?")
print("=" * 90)
bl = pd.read_csv(DIR + r"\servo_analysis_precision_20260708_122810.csv")
joints = ["servo6", "servo7", "servo8", "servo9"]
hold = bl[bl["phase"] == "HOLD_CLOSED"].copy()
hold["rstep"] = hold.groupby("cycle").cumcount()
for j in joints:
    piv = hold.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
    m = piv.mean(axis=1)
    line = "  ".join(f"h{int(k)}:{v:+.4f}" for k, v in m.iloc[:15].items())
    print(f"{j}: {line}")
    tail = piv.iloc[15:]
    print(f"   nach Settle (Steps 15+): mean={tail.values.mean():+.4f} "
          f"std={tail.values.std():.4f}")

# ------------------- C. Slow-Close: Tracking-Fehler bei delta_norm=0.003
print()
print("=" * 90)
print("C. SLOW CLOSE (delta_norm=0.003) vs NORMAL (0.005): Tracking-q_delta im Vergleich")
print("=" * 90)
slow = pd.read_csv(DIR + r"\servo_analysis__temp_slow_20260708_133732.csv")
norm = pd.read_csv(DIR + r"\servo_analysis_precision_20260708_133601.csv")
for name, d in [("slow(0.003)", slow), ("normal(0.005)", norm)]:
    dc = d[d["phase"] == "CLOSE"].copy()
    dc["rstep"] = dc.groupby("cycle").cumcount()
    dc = dc[dc["rstep"] >= 25]  # Startup raus
    for j in ["servo6", "servo8"]:
        col = dc[f"{j}_q_delta"]
        piv = dc.pivot_table(index="rstep", columns="cycle", values=f"{j}_q_delta")
        cyc_std = piv.std(axis=1).median()
        print(f"{name} {j}: mean={col.mean():+.4f} std={col.std():.4f} "
              f"max={col.max():+.4f}  Zyklus-zu-Zyklus-Std={cyc_std:.4f}")

# ------------------- D. Startup-Transient: Dauer und Variabilitaet
print()
print("=" * 90)
print("D. STARTUP: Step, an dem q_measured erstmals > 0.02 (Servo beginnt zu fahren)")
print("=" * 90)
blc = bl[bl["phase"] == "CLOSE"].copy()
blc["rstep"] = blc.groupby("cycle").cumcount()
for j in joints:
    starts = []
    for c, grp in blc.groupby("cycle"):
        grp = grp.sort_values("rstep")
        moved = grp[grp[f"{j}_q_measured"] > 0.02]
        starts.append(int(moved["rstep"].min()) if len(moved) else None)
    print(f"{j}: Start-Steps pro Zyklus: {starts}")

# ------------------- E. Max q_delta waehrend Startup (das ist der stoerende Peak)
print()
print("Max q_delta in Steps 0-25 (Startup-Peak, pro Freilauf-Zyklus):")
for j in joints:
    peaks = []
    for c, grp in blc.groupby("cycle"):
        peaks.append(round(float(grp[grp["rstep"] <= 25][f"{j}_q_delta"].max()), 3))
    print(f"  {j}: {peaks}")
