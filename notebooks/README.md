# Notebooks and scripts

Validation notebooks of `panels` against results of the literature, the
utilities they share, and the scripts of the studies of the cohesive zone
model. The notebooks need `panels` (`pip install panels`, or the repository
built in place with `python setup.py build_ext --inplace` and on the
`PYTHONPATH`) and are run from this folder. The scripts are run from the root
of the repository and put the repository first on `sys.path`.

Most notebooks load stored results from `results/` and only run `panels`
again with `RERUN = True` in their first cell. The stored results of the
notebooks were computed with `panels` 0.14.0, `structsolve` 0.9.1 (0.9.2
for `sze2004_hinged_roofs.ipynb`) and `composites` 0.10.0; the results
computed again keep the run times of their first computation, except the
DCB results, `results/*.npz`, and those of `conn_method_comparison.ipynb`,
whose run times were measured again with `panels` 0.14.0 and `structsolve`
0.9.2, one run at a time on an idle machine, with both connection methods
under the same conditions.

## Mode I double cantilever beam (DCB), cohesive zone model

The cohesive zone model (`'SB_TSL'` connection) is described in
`doc/source/cohesive_zone.rst`, with the convergence study and the validation
against the literature.

| File | Content |
|---|---|
| `dcb_utils.py` | Multidomain model of the DCB (`build_dcb`), nonlinear solution under prescribed openings (`solve_dcb`) with the check of the equilibrium between the load and the cohesive tractions (`equilibrium_check`), stored results (`run_or_load`, `load`), rules of the discretisation (`resolution`, `discretize`, `margin`) and references of linear elastic fracture mechanics (beam theory) |
| `dcb_alfano2001.ipynb` | Alfano and Crisfield (2001): DCB of Table I with the nominal and the low strength, XAS-913C DCB of Table II |
| `dcb_camanho2003.ipynb` | Camanho, Dávila and de Moura (2003): AS4/PEEK DCB, kept for reference only, see the note in the notebook |
| `dcb_turon2007.ipynb` | Turon et al. (2007): T300/977-2 DCB, with the sensitivity to the penalty stiffness |
| `dcb_krueger2008.ipynb` | Krueger (2008), NASA/TM-2008-215123: unidirectional and multidirectional DCB benchmarks (VCCT) |
| `dcb_tijs2022.ipynb` | Tijs et al. (2022): AS4D/PEKK-FC DCB, the reference DCB of the convergence study |
| `dcb_tijs2023_phd.ipynb` | Tijs (2023), PhD thesis: compliance calibration and conduction-welded DCB |
| `dcb_lecinana2023.ipynb` | Leciñana et al. (2023): initial compliance of AS4D/PEKK-FC DCB specimens (linear analysis) |
| `dcb_mode_I_validation_summary.ipynb` | Table of the peak loads of the notebooks above |
| `krueger2012_mmb_benchmark.ipynb` | Krueger (2012), NASA/CR-2012-217562: mixed-mode bending benchmark data, recorded but not run, the damage of `'SB_TSL'` being driven by the normal separation only |
| `dcb_mode_I_convergence_study.py` | Convergence study of the discretisation of the cohesive domain on the reference DCB, results in `results/dcb_mode_I_convergence_study.json` |
| `dcb_mode_I_solver_study.py` | Solution procedure on the reference DCB: consistent tangent against the secant stiffness, with and without line search and bisection |
| `dcb_mode_I_validation_rerun.py` | Runs the nonlinear runs of the DCB notebooks again, in parallel, overwriting `results/*.npz` |

The docstring of each script gives its usage. The scripts write their runs to
folders at the root of the repository (`dcb_mode_I_convergence_study_runs`,
`dcb_mode_I_solver_study_runs`, `dcb_mode_I_validation_rerun_runs`), which are
not part of the repository; only the results used by the documentation are
kept, in `results/`.

## Penalty and null-space connections

| File | Content |
|---|---|
| `conn_method_comparison.ipynb` | The benchmark notebooks with the penalty and the null-space connections |
| `conn_method_study.py` | Runs and timings of `conn_method_comparison.ipynb`: null-space runs of the DCB cases of the notebooks, elastic compliances, profile of the cost and Stamatelos and Labeas models with both methods, results in `results/conn_methods/`; the penalty results of the DCB cases are those of the DCB notebooks, `results/*.npz` |

## Shells and plates

| File | Content |
|---|---|
| `pagano1970_plates.ipynb`, `pagano1970.py` | Pagano (1970): cross-ply and sandwich plates under a sinusoidal pressure, with the exact 3D elasticity solution |
| `varadan1991_cylinders.ipynb` | Varadan and Bhaskar (1991): cross-ply cylinders under a sinusoidal load |
| `noor1975_buckling.ipynb` | Noor (1975): buckling of cross-ply plates in uniaxial compression |
| `kardomateas1993_thick_rings.ipynb` | Kardomateas (1993): thick orthotropic cylinders under external pressure |
| `kardomateas1993b_1995_thick_cylinders.ipynb` | Kardomateas (1993b), Kardomateas and Philobos (1995): thick cylinders in axial compression and under combined loads |
| `kardomateas2003_sandwich_rings.ipynb` | Kardomateas and Simitses (2003): long sandwich cylinders under hydrostatic pressure |
| `han2004_sandwich_rings.ipynb` | Han, Kardomateas and Simitses (2004): long sandwich cylinders under hydrostatic pressure |
| `nasa_sp8007_external_pressure.ipynb` | NASA/SP-8007-2020/REV 2: cylinders under lateral and hydrostatic pressure |
| `schweizerhof1984_pressure_loads.ipynb` | Schweizerhof and Ramm (1984): displacement dependent pressure loads |
| `sze2004_hinged_roofs.ipynb` | Sabir and Lock (1972), Sze et al. (2004): limit loads of hinged cylindrical roofs |
| `stamatelos_labeas_2023.ipynb`, `stamatelos_labeas_2023_multidomain.ipynb` | Stamatelos and Labeas (2023): buckling of laminated stiffened plates, single domain and multidomain |
| `follower_utils.py` | Utilities of the follower (hydrostatic) pressure notebooks |
| `scf_validation_utils.py` | Utilities of the shear correction notebooks |

## Quadrature

`test_quadrature_partial_integration_matrix.ipynb` and
`test_quadrature_partial_integration_of_domain.ipynb` check the partial
integration of polynomials with Gauss-Legendre rules, taking the points and
weights from `scipy.special.roots_legendre`. They do not use `panels`.

## Results

`results/` holds the stored results of the notebooks and of the convergence
study. The DCB results, `results/*.npz`, are read with `dcb_utils.load`:
besides the load-opening curves, the iterations and the run times, they store
the model and its discretisation (`case`), the solver settings (`solver`),
the version of `panels` and the connection method of the run.
