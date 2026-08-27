# Foto-Sweep mit langen Haltezeiten: faehrt q = 0, 0.25, 0.5, 0.75, 1.0
# nacheinander an und haelt jeden Punkt lange genug zum Fotografieren.
# Stellung exakt wie eval/qsweep_check.py.
import sys, time
from hardware.ar10 import AR10Interface
from sim.hand import CONTROL_JOINTS, SERVO0_INIT

port  = sys.argv[1]
hold  = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
first = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
JOINTS = ["servo2", "servo3", "servo4", "servo5", "servo6", "servo7", "servo8", "servo9"]
STEPS  = [0.0, 0.25, 0.5, 0.75, 1.0]

def targets_for(q):
    out = [0.0] * len(CONTROL_JOINTS)
    out[0] = SERVO0_INIT
    for j in JOINTS:
        out[CONTROL_JOINTS.index(j)] = q
    return out

ar10 = AR10Interface(com_port=port)
t0 = time.time()
try:
    for i, q in enumerate(STEPS):
        ar10.send_q_target(targets_for(q))
        wait = first if i == 0 else hold
        time.sleep(4.0)                      # anfahren + settlen
        meas = [sum(v) / 5 for v in zip(*[ar10.read_q_measured() for _ in range(5)])]
        err  = max(abs(meas[CONTROL_JOINTS.index(j)] - q) for j in JOINTS)
        print("[%5.1fs] q=%.2f STEHT  (Restfehler %.3f) -- jetzt fotografieren, "
              "naechster Punkt in %.0f s" % (time.time() - t0, q, err, wait - 4.0), flush=True)
        time.sleep(max(0.0, wait - 4.0))
    print("\n[%5.1fs] Sweep durch. Hand faehrt auf offen zurueck." % (time.time() - t0), flush=True)
    ar10.send_q_target(targets_for(0.0))
    time.sleep(3.0)
finally:
    ar10.close()
