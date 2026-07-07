from __future__ import annotations


def step_reward(
    n_contact_prev:   int,
    n_contact_now:    int,
    n_target:         int,
    pedestal_contact: bool,
    gamma:            float,
    cfg:              dict,
    terminal:         bool = False,
) -> float:
    # Zeit-Penalty + Potential-Based Reward Shaping (Ng, Harada & Russell 1999).
    #
    # Shaping-Term F = gamma * Phi(s') - Phi(s) mit dem Potential
    #     Phi(s) = w_contact * min(n_contact, n_target) / n_target.
    # PBRS ist die EINZIGE Shaping-Form, die beweisbar die optimale Policy nicht
    # veraendert (policy-invariant). Anders als ein naiver "Bonus pro neuem
    # Kontakt" bestraft es Kontaktverlust symmetrisch -> kein Toggle-Farming
    # (Finger an-/abtippen netto ~0 statt netto positiv).
    #
    # terminal=True: der Folgezustand ist der absorbierende Endzustand, dessen
    # Potential per Definition 0 ist (Phi(terminal)=0). Notwendig, damit die
    # Invarianz-Garantie in episodischen Tasks exakt gilt.
    phi_prev = cfg["w_contact"] * min(n_contact_prev, n_target) / n_target
    phi_now  = 0.0 if terminal else cfg["w_contact"] * min(n_contact_now, n_target) / n_target

    r  = cfg["r_step"]
    r += gamma * phi_now - phi_prev
    if pedestal_contact:
        r -= cfg["w_pedestal"]
    return r


def terminal_reward(lifted: bool, cfg: dict) -> float:
    # Wird einmalig am Episodenende aufgerufen, nach dem Lift-Test.
    # Das ist das eigentliche Aufgaben-Ziel - PBRS-Shaping aendert daran nichts.
    return cfg["r_lift_success"] if lifted else cfg["r_lift_fail"]
