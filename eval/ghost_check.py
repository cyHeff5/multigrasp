"""Geistermaterial der Benchmark-Kollisionsgeometrie messen.

Punkte auf der KOLLISIONSHUELLE sampeln und den Abstand zum echten (visual-)
Mesh messen. Das ist der Abstand, mit dem ein Finger "greift", ohne das Objekt
zu beruehren — die Groesse aus SENSOR_ANALYSIS_FINDINGS.md 9.0.

Zwei Fallen, beide hier vermieden (25.08. abends teuer gelernt):
  * Die Gegenrichtung (Mesh-Punkt -> Tiefe in der Huelle) unterschaetzt
    Konkavitaeten massiv und meldet fuer Teil 10 nur 0.5 statt 10 mm.
  * Als Referenz die Mesh-VERTICES zu nehmen ist bei groben Dreiecken (ein
    Quader hat 12) unbrauchbar: die Flaechenmitte liegt zentimeterweit vom
    naechsten Vertex weg. Referenz ist deshalb eine dichte Punktwolke auf der
    Mesh-Oberflaeche.
Ein direkter PyBullet-Test per getClosestPoints geht auch, hat aber einen
konstanten Sockel von exakt 1.00 mm (Bullet-Margin fuer konvexe Mesh-Shapes,
an einer perfekten Box nachgemessen) — hier bewusst mesh-basiert.

Aufruf:  ../sim2real/.venv/bin/python -m eval.ghost_check
"""
import sys, struct, xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from assets import benchmark_part_urdf

ASSETS = Path(__file__).resolve().parent.parent / "assets"
N = 6000

def read_stl(path):
    data = path.read_bytes(); n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data, dtype=np.uint8, offset=84)[: n*50].reshape(n,50)
    return rec[:,12:48].copy().view("<f4").reshape(n,3,3).astype(np.float64)

def read_obj_tris(path):
    V, F = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):  V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): F.append([int(t.split("/")[0]) for t in line.split()[1:4]])
    V = np.array(V); F = np.array(F) - 1
    return V[F]

def sample(tris, n, rng):
    a,b,c = tris[:,0], tris[:,1], tris[:,2]
    area = 0.5*np.linalg.norm(np.cross(b-a, c-a), axis=1)
    if area.sum() <= 0: return np.empty((0,3))
    idx = rng.choice(len(tris), size=n, p=area/area.sum())
    u,v = rng.random(n), rng.random(n); f = u+v>1; u[f],v[f] = 1-u[f],1-v[f]
    t = tris[idx]
    return t[:,0] + u[:,None]*(t[:,1]-t[:,0]) + v[:,None]*(t[:,2]-t[:,0])

def parts_of(urdf):
    root = ET.parse(urdf).getroot()
    vis = root.find(".//link/visual"); vm = vis.find("geometry/mesh")
    vs  = float(vm.get("scale","1 1 1").split()[0])
    cols = []
    for c in root.findall(".//link/collision"):
        m = c.find("geometry/mesh")
        cols.append((ASSETS/"benchmark_parts"/m.get("filename"), float(m.get("scale","1 1 1").split()[0])))
    return (ASSETS/"benchmark_parts"/vm.get("filename"), vs), cols

rng = np.random.default_rng(0)
print(f"{'Teil':>4} {'Huellen':>7} {'ghost p95':>10} {'ghost max':>10} {'>2mm':>7} {'>5mm':>7}")
rows = []
for pid in range(1, 15):
    (vpath, vs), cols = parts_of(benchmark_part_urdf(pid))
    mesh_t = read_stl(vpath)*vs
    # Dichte Punktwolke auf der Mesh-OBERFLAECHE statt der Vertices: bei groben
    # Dreiecken (Quader = 12 Dreiecke) liegt die Flaechenmitte zentimeterweit
    # vom naechsten Vertex entfernt, das ergibt Phantom-Ghost.
    dense = np.vstack([mesh_t.reshape(-1,3), sample(mesh_t, 400_000, rng)])
    tree = cKDTree(dense)
    pts = []
    for cpath, cs in cols:
        t = read_obj_tris(cpath)*cs
        pts.append(sample(t, max(200, N//len(cols)), rng))
    pts = np.vstack([q for q in pts if len(q)])
    d, _ = tree.query(pts)
    d = d*1000.0
    print(f"{pid:>4} {len(cols):>7} {np.percentile(d,95):>8.2f}mm {d.max():>8.2f}mm "
          f"{100*(d>2).mean():>6.1f}% {100*(d>5).mean():>6.1f}%")
    rows.append((np.percentile(d,95), pid))
print("\nnach p95:", [(f"P{p}", f"{s:.1f}mm") for s,p in sorted(rows, reverse=True)])
