"""G2's real chance (200 sampled playouts from the true position, its own move kept) vs its value, through the game."""
import sys
sys.argv = ['x', 'game_297763546.json']
import rollouts as R
from multiprocessing import Pool
if __name__ == '__main__':
    idx = [2, 46, 57, 83, 103, 107, 117, 121, 126]
    C = R.G['choices']
    idx = [next(j for j in range(i, len(C)) if C[j][0] == 2) for i in idx]
    with Pool(12) as P:
        res = R.run(P, [(i, C[i][1]) for i in idx], 200, 90_000_000)
    for i in idx:
        g = R.replay(C[:i]); v = R.ev(g.observe())[1]; t = g.view(1, 1 << 40)['turn']['number']
        print(f'#{i:3d} turn {t:2d} G2 says {(v + 1) / 2:.2f}  playouts {R.wil(*res[(i, C[i][1])])}   {R.options(g)[C[i][1]][:60]}')
