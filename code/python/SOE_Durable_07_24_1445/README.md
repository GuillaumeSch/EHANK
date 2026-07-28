# E-HANK — package état au 24 juillet 2026

## Installation

```bash
pip install sequence-jacobian==1.0.0 numba
```

`numba` doit être installé explicitement : SSJ ne le tire pas comme dépendance
et les `StageBlock` sont ~50x plus lents sans lui.

## Arborescence

```
ehank_package/
├── blocks.py        blocs agrégés + blocs numéraire (numeraire_cpi / numeraire_core)
├── household.py     ménage StageBlock, générique en numéraire
├── calibration.py   BASE / DURABLE / POLICY + make_calibration
├── model.py         assemblage DAG, SS, chocs, runner d'expériences
├── deflator.py      déflateur pondéré par les parts, hors DAG   [NOUVEAU]
├── tests.py         non-régression checks 2-5                   [NOUVEAU]
├── welfare.py       CEV
├── plotting.py      helpers figures
├── run_experiments.py       E1-E5 + E8 (écart de déflateur)
├── run_dose_response.py     E6-E7
└── run_linearity_check.py   E9 (diagnostic linéaire vs non-linéaire) [NOUVEAU]
```

## Ordre d'exécution

```bash
python run_experiments.py        # ~15-25 min, 14 IRF, cache repris si interrompu
python run_dose_response.py      # ~15 min
python run_linearity_check.py    # lent (non-linéaire), lancer en dernier
```

Les trois écrivent dans `output/`. Les caches sont dans `cache_core/` et
`cache_dose_core/` ; ils sont repris automatiquement, donc une exécution
interrompue redémarre là où elle s'est arrêtée.

## Numéraire

Deux numéraires sont implémentés et testés. Le défaut est **`core`**.

| | `'cpi'` | `'core'` |
|---|---|---|
| unité de compte | panier IPC (ARS) | bien domestique |
| `p_num` | 1 (constante de calibration) | `pH_P` (sortie de bloc) |
| `r_num` | `r` | `(1+r)·pH_P(-1)/pH_P − 1` |

Le ménage ne référence jamais l'IPC : il lit `p_num`, `r_num`, `atw_n_num` et
renvoie `A` en unités de compte, converti par `assets_convert` en `A_cpi` avant
`assets_clearing` et `CA`.

Pour basculer : `build_model('cpi')` + `make_calibration('cpi', ...)` +
`run(..., numeraire='cpi')`. **Le calibrage et le modèle doivent utiliser le
même numéraire** — sous `'cpi'` la calibration fournit `p_num = 1` comme
constante, sous `'core'` elle doit ne pas contenir `p_num` du tout.

Les caches sont taggés par numéraire (`cache_core/` vs `cache_cpi/`). Ne jamais
réutiliser un cache d'un numéraire à l'autre : l'écart est de ~0.03 %,
invisible à l'œil mais suffisant pour polluer un tableau.

## Prix énergie

| variable | signification |
|---|---|
| `pE_P` | prix de marché gros/détail, avant subvention |
| `pE_B_P` | `P_B^E`, prix payé par un ménage brun = `(1−τE)·pE_P + τE·pE_P.ss` |
| `pE_G_P` | `P_G^E`, prix payé par un adoptant vert |

`pEhh_P` n'existe plus.

## Choix de modélisation : ancrage de l'IPC (Option C)

Le volet énergie de l'IPC est ancré sur `P_B^E` seul, pas sur l'indice pondéré
par les parts. Justification dans le docstring de `CESprices`. L'écart de mesure
est chiffré ex post par `ehank/deflator.py`, sans rétroaction dans le DAG :

```
phi^(1-eta_E) = 1 + alpha_E * D_G * (P_G^(1-eta_E) - P_B^(1-eta_E))
```

Forme fermée exacte (pas de `pHF_P`) parce que `inner_nest` impose
`alpha_E·P_B^(1-eta_E) + (1-alpha_E)·pHF_P^(1-eta_E) = 1` à chaque date.

`run_experiments.py` produit `output/deflator_table_core.txt` et
`output/fig8_deflator_gap.png`.

## Non-régression

```python
from model import build_model, solve_ss
from calibration import make_calibration
import tests as T

M = build_model('core')
solve = lambda c: solve_ss(M, c)
mk = lambda **kw: make_calibration('core', **kw)
ss = solve(mk())
T.run_all(ss, solve=solve, make=mk, label='core', full=True)   # full=True est lent
```

Valeurs de référence (n_a=150, sous les deux numéraires, identiques) :

| check | valeur |
|---|---|
| `test_targets(ss)` | PASS (`assets_clearing` 6.7e-07) |
| enveloppe, médiane | 2.54e-04 |
| enveloppe, ordre de convergence | ratios 2.41 / 2.59 sur n_a 150→300→600 |
| monotonicité `a'` | PASS |
| stock-flux | −2.9e-10 |
| phase 2 (canal fermé) | PASS, 3.9e-10 |

`check_envelope_order` et `check_phase2` prennent une **fabrique** de
calibration, pas un dict : `cE_ss_grid` est de forme `(n_e, n_a)` et doit être
reconstruit quand la grille change.

## Points ouverts

1. **`taste_shock` et `psi_g` ne sont pas identifiés séparément** par une seule
   cible de part d'adoption. Il faut un moment d'élasticité issu de la
   littérature empirique. Inchangé depuis les sessions précédentes.

## Écarts CPI vs core (choc prix, n_a=150) — pour mémoire

| var | max\|IRF\| | max\|écart\| | rel |
|---|---|---|---|
| y | 1.04e-02 | 1.49e-06 | 0.01 % |
| C | 2.19e-02 | 2.33e-06 | 0.01 % |
| D_GREEN | 3.19e-01 | 5.79e-05 | 0.02 % |

`y` et `C` : bruit de différences finies (minimum en U autour de h=1e-4, le pas
hardcodé de `StageBlock.preliminary_hetinput`). `D_GREEN` : discrétisation de
la grille, 5.79e-05 → 3.02e-05 quand `n_a` passe de 150 à 300 (O(1/n_a)).
Le résidu de 4 % de l'implémentation précédente n'existe plus.
