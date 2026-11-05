"""Task 4 camera-loop arms side by side (results/v1_1/t4_camera): per episode the rest state of every pushed block
(1 flat clear, T tipped, L on/leaning on base, t touching base, B on a block, O off the table), the push sequence
(R = the recovery node's re-push, ! = the node failed) and per-arm tallies of the pushed blocks' rest states.

  python3 scripts/t4_camera_cmp.py <variant 4-2|4-3> <label>=<arm dir> [<label>=<arm dir> ...] [--episodes]
"""
import collections
import glob
import json
import re
import sys

R = '/home/visakii/Documents/tower_rl/results/v1_1/t4_camera'
VAR = {'4-2': 'task4-2_full_tower_pushdown_2', '4-3': 'task4-3_full_tower_pushdown_3'}
ABBR = {'flat clear': '1', 'tipped': 'T', 'on/leaning on base': 'L', 'touching base': 't', 'on a block': 'B',
        'off the table': 'O'}


def load(arm, var):
    f, = glob.glob(f'{R}/{arm}/{var}/*/episodes.json')
    return {e['extra']['case']: e for e in json.load(open(f))}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    v, arms = args[0], [a.split('=', 1) for a in args[1:]]
    data = {lab: load(arm, VAR[v]) for lab, arm in arms}
    if '--episodes' in sys.argv:
        for case in data[arms[0][0]]:
            print(f'--- {case}')
            for lab, _ in arms:
                e = data[lab][case]
                x = e['extra']
                rest = ''.join(ABBR.get(r[0], '?') for r in x['rest'])
                seq = []
                for t, name, res, info in x['policy_log']:
                    m = re.match(r'p(\d+)_(re_)?push$', name)
                    if m:
                        seq.append(('R' if m.group(2) else '') + m.group(1) + ('' if res == 'ok' else '!'))
                print(f'  {lab:10s} TSR {int(e["task_success"])} rest {rest:10s} broken {int(x["structure_broken"])} '
                      f'seq {" ".join(seq)}')
    print(f'{"arm":10s} {"TSR":>5s} {"blocks":>6s} ' + ' '.join(f'{k:>4s}' for k in ABBR.values())
          + '  push-node failures (why)')
    for lab, _ in arms:
        c = collections.Counter()
        why = collections.Counter()
        tsr = 0
        for e in data[lab].values():
            tsr += int(e['task_success'])
            for r in e['extra']['rest']:
                c[ABBR.get(r[0], '?')] += 1
            for t, name, res, info in e['extra']['policy_log']:
                if re.match(r'p\d+_(re_)?push$', name) and res != 'ok':
                    why[f'{res}:{info.get("why", "")}'] += 1
        n = sum(c.values())
        print(f'{lab:10s} {tsr:>2d}/{len(data[lab]):<2d} {n:>6d} ' + ' '.join(f'{c[k]:>4d}' for k in ABBR.values())
              + '  ' + ', '.join(f'{k} {n_}' for k, n_ in why.most_common()))


if __name__ == '__main__':
    main()
