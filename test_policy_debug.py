from stable_baselines3 import PPO
import numpy as np

m = PPO.load("artifacts/models/precision/seed_0_dr/best/best_model")
print("obs_space:", m.observation_space)
print("act_space:", m.action_space)

for obs in [
    [0,0,0,0], [0,0,0.1,0.1], [0,0,0.3,0.3], [0,0,0.5,0.5],
    [1,1,0.3,0.3], [1,0,0.5,0.3],
]:
    o = np.array(obs, dtype=np.float32)
    a, _ = m.predict(o, deterministic=True)
    print(f"obs={obs} -> action={a}")
