# Faehrt die Hand auf einen einzelnen q-Punkt und LAESST SIE DORT STEHEN.
# Der Maestro haelt das Target auch nachdem das Skript beendet ist -- damit
# kann in Ruhe fotografiert werden, statt einem durchlaufenden Sweep
# hinterherzuhetzen. Stellung exakt wie eval/qsweep_check.py (servo0 auf
# SERVO0_INIT, servo1 offen, servo2..servo9 auf q).
import sys, time
from hardware.ar10 import AR10Interface
from sim.hand import CONTROL_JOINTS, SERVO0_INIT

port = sys.argv[1]
q    = float(sys.argv[2])
JOINTS = ["servo2", "servo3", "servo4", "servo5", "servo6", "servo7", "servo8", "servo9"]

targets = [0.0] * len(CONTROL_JOINTS)
targets[0] = SERVO0_INIT
for j in JOINTS:
    targets[CONTROL_JOINTS.index(j)] = q

ar10 = AR10Interface(com_port=port)
ar10.send_q_target(targets)
time.sleep(6.0)
meas = [sum(v) / 5 for v in zip(*[ar10.read_q_measured() for _ in range(5)])]
print("q_soll = %.2f   Hand steht und haelt." % q)
print("  " + "  ".join("%s:%.3f" % (j, meas[CONTROL_JOINTS.index(j)]) for j in JOINTS))
err = max(abs(meas[CONTROL_JOINTS.index(j)] - q) for j in JOINTS)
print("  groesster Restfehler: %.3f" % err)
ar10.close()
