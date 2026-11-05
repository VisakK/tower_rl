"""Task 4 camera loop: at the row where each ok push's estimate first passed the departure test (push.departed: r >=
0.16 m, or r >= 0.09 m and z <= 0.02 m), was the true block departed? Confidence of that row; final rest state."""
import json, glob, pickle, re, collections
import numpy as np
import sys
R = '/home/visakii/Documents/tower_rl'
INCLUDE_RE = '--re' in sys.argv  # also the recovery node's re-pushes
ARMS = ['settle/off', 'settle/on', 'v2_r1', 'v2_r1_repush2']
def dep(p, txy):
    r = np.linalg.norm(p[:2] - txy)
    return r >= 0.16 or (r >= 0.09 and p[2] <= 0.02), r
tab = collections.Counter(); ex = []
for var in ['task4-2_full_tower_pushdown_2', 'task4-3_full_tower_pushdown_3']:
    g = json.load(open(f'{R}/results/bundles/student_v1_1/graphs/{var}.json'))
    blk = {n['id']: n['block'] for n in g['nodes'] if n.get('skill') == 'push'}
    order = [n['block'] for n in g['nodes'] if n.get('skill') == 'push']
    found = g['tower']['foundation']
    for arm in ARMS:
        d, = glob.glob(f'{R}/results/v1_1/t4_camera/{arm}/{var}/*/')
        eps = json.load(open(d + 'episodes.json'))
        pl = pickle.load(open(d + 'perception_log.pkl', 'rb')); ids = pl['block_ids']
        by_seed = collections.defaultdict(list)
        for e in pl['log']: by_seed[e['seed']].append(e)
        for ep in eps:
            x = ep['extra']; L = by_seed[x['seed']]
            txy = np.mean([L[0]['true_pos'][ids.index(f)][:2] for f in found], 0)
            t_prev = 0.0
            for t, name, res, info in x['policy_log']:
                m = re.match(r"(p\d+)_(re_)?push$", name)
                if m and res == 'ok' and (INCLUDE_RE or not m.group(2)):
                    b = blk[m.group(1) + '_push']; i = ids.index(b)
                    win = [e for e in L if t_prev < e['t'] <= t + 1e-6]
                    hit = next((e for e in win if dep(e['pos'][i], txy)[0]), None)
                    rest = x['rest'][order.index(b)][0]
                    if hit is None:
                        tab[(var[5:8], 'no est. departure in window', rest)] += 1
                    else:
                        td, tr = dep(hit['true_pos'][i], txy); _, er = dep(hit['pos'][i], txy)
                        c = 'conf' if hit['conf'][i] >= 0.5 else 'unconf'
                        tab[(var[5:8], c, 'true departed' if td else 'TRUE NOT departed', rest)] += 1
                        if not td: ex.append((var[5:8], arm, x['case'], name, hit['t'], round(float(hit['conf'][i]), 1), round(tr*1000), round(er*1000), rest))
                t_prev = t
for k in sorted(tab): print(k, tab[k])
print(len(ex)); [print(e) for e in ex[:60]]
