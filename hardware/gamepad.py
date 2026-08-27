# Gamepad- und Tastatur-Input.
# read_inputs() gibt immer dasselbe Dict zurück.
#
# Ohne Gamepad wird die Tastatur gelesen. Dafür gibt es zwei Wege, in dieser
# Reihenfolge: das PyBullet-GUI-Fenster (drive_pregrasp/drive_calibration haben
# ohnehin eins, dort tippt man also direkt in die Szene), sonst ein kleines
# pygame-Fenster. Der pygame-Weg braucht zwingend ein Fenster — ohne
# display.set_mode() liefert key.get_pressed() grundsätzlich nichts, egal was
# gedrückt wird.
from __future__ import annotations

# Totzone für Analogsticks: Werte darunter werden als 0 behandelt (Drift-Unterdrückung).
_DEAD = 0.08


def _btn(js, i: int) -> bool:
    try:
        return bool(js.get_numbuttons() > i and js.get_button(i))
    except Exception:
        return False


def _axis(js, i: int) -> float:
    try:
        v = float(js.get_axis(i))
        return v if abs(v) > _DEAD else 0.0
    except Exception:
        return 0.0


# Tastenbelegung des Tastatur-Fallbacks, an einer Stelle, damit PyBullet-Pfad,
# pygame-Pfad und die Hilfeausgabe nicht auseinanderlaufen.
KEYMAP_HELP = ("SPACE=A  BACKSPACE=B  X=X  Y=Y  A=LB  D=RB  ESC=Menu  "
               "Pfeiltasten=Stick")


def _pybullet_inputs() -> dict | None:
    # Tastatur aus dem PyBullet-GUI-Fenster. None -> keine GUI da, Aufrufer
    # faellt auf pygame zurueck.
    try:
        import pybullet as p
        info = p.getConnectionInfo()
        if not info.get("isConnected") or info.get("connectionMethod") != p.GUI:
            return None
        ev = p.getKeyboardEvents()
    except Exception:
        return None

    def down(code: int) -> bool:
        return bool(ev.get(code, 0) & p.KEY_IS_DOWN)

    return {
        "a":    down(p.B3G_SPACE),
        "b":    down(p.B3G_BACKSPACE),
        "x":    down(ord("x")),
        "y":    down(ord("y")),
        "rb":   down(ord("d")),
        "lb":   down(ord("a")),
        "menu": down(27),                       # ESC, PyBullet hat dafuer keine Konstante
        "sx":   (-1.0 if down(p.B3G_LEFT_ARROW) else (1.0 if down(p.B3G_RIGHT_ARROW) else 0.0)),
        "sy":   (-1.0 if down(p.B3G_DOWN_ARROW) else (1.0 if down(p.B3G_UP_ARROW)    else 0.0)),
    }


def read_inputs(js) -> dict:
    # js=None -> Tastatur-Fallback (Belegung siehe KEYMAP_HELP).
    if js is None:
        inp = _pybullet_inputs()
        if inp is not None:
            return inp
    import pygame
    if js is None and pygame.display.get_surface() is None:
        # Ohne Surface liefert key.get_pressed() nichts. Hier erst anlegen, weil
        # init_pygame_joystick() vor init_pybullet() laeuft und dort noch nicht
        # entschieden ist, ob es ein PyBullet-Fenster geben wird.
        pygame.display.set_mode((420, 120))
        pygame.display.set_caption("multigrasp Tastatursteuerung — Fenster fokussiert lassen")
        print(f"[gamepad] Tastatur-Fenster geoeffnet. {KEYMAP_HELP}")
    pygame.event.pump()
    if js is not None:
        return {
            "a":    _btn(js, 0),
            "b":    _btn(js, 1),
            "x":    _btn(js, 2),
            "y":    _btn(js, 3),
            "rb":   _btn(js, 5),
            "lb":   _btn(js, 4),
            "menu": _btn(js, 7),
            "sx":   _axis(js, 0),
            "sy":  -_axis(js, 1),  # Joystick-Konvention: oben = negativ -> invertieren.
        }
    else:
        keys = pygame.key.get_pressed()
        return {
            "a":    bool(keys[pygame.K_SPACE]),
            "b":    bool(keys[pygame.K_BACKSPACE]),
            "x":    bool(keys[pygame.K_x]),
            "y":    bool(keys[pygame.K_y]),
            "rb":   bool(keys[pygame.K_d]),
            "lb":   bool(keys[pygame.K_a]),
            "menu": bool(keys[pygame.K_ESCAPE]),
            "sx":   (-1.0 if keys[pygame.K_LEFT]  else (1.0 if keys[pygame.K_RIGHT] else 0.0)),
            "sy":   (-1.0 if keys[pygame.K_DOWN]  else (1.0 if keys[pygame.K_UP]    else 0.0)),
        }


def init_pygame_joystick():
    # Gibt den ersten gefundenen Joystick zurück, oder None wenn keiner angeschlossen ist.
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[gamepad] {js.get_name()}")
        return js
    print("[gamepad] Kein Gamepad gefunden — Tastatur-Fallback aktiv.")
    print(f"[gamepad] {KEYMAP_HELP}")
    print("[gamepad] Tasten gehen an das PyBullet-Fenster, sofern eins laeuft;")
    print("[gamepad] sonst oeffnet sich ein eigenes Fenster, das den Fokus braucht.")
    return None
