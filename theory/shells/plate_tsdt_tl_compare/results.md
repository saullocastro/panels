# Output of `compare.py`, hpc12 job 1059815 (2026-11-11)

```
1. rigid rotation of a free plate a/h = 6, |fint|/|KC0 c|
axis, angle: plate_tsdt_tlA_proto, plate_tsdt_tlB_proto, plate_fsdt_tl, plate_tsdt_gl
(1, 0, 0), 0.2: 1.247e-04, 9.833e-05, 1.353e-04, 2.324e-14
(1, 0, 0), 0.1: 1.553e-05, 1.238e-05, 1.703e-05, 3.675e-14
(0, 1, 0), 0.2: 9.251e-05, 7.293e-05, 9.393e-05, 3.323e-14
(0, 1, 0), 0.1: 1.152e-05, 9.180e-06, 1.182e-05, 6.193e-14
(1, 2, 3), 0.2: 8.424e-03, 4.250e-06, 5.097e-06, 3.283e-02
(1, 2, 3), 0.1: 8.262e-03, 5.253e-07, 6.287e-07, 3.283e-02

2. thin strip L = 1, b = 0.04, h = 2e-3, nu = 0, 14 x 4 terms, vs elastica
alpha = 1, tip rotation 0.461 rad, elastica w = 0.3017208 L, -u = 0.0564332 L
  plate_tsdt_tlA_proto   w = 0.2246729 (-25.536 %), -u = 0.0300923 (-46.676 %), 5.7 s
  plate_tsdt_tlB_proto   w = 0.3010715 (-0.215 %), -u = 0.0561673 (-0.471 %), 6.5 s
  plate_fsdt_tl          w = 0.3009113 (-0.268 %), -u = 0.0561018 (-0.587 %), 5.1 s
  plate_tsdt_gl          w = 0.3333342 (+10.478 %), -u = 0.0702290 (+24.446 %), 2.8 s
alpha = 3, tip rotation 0.986 rad, elastica w = 0.6032534 L, -u = 0.2544202 L
  plate_tsdt_tlA_proto   w = 0.3025068 (-49.854 %), -u = 0.0525299 (-79.353 %), 9.7 s
  plate_tsdt_tlB_proto   w = 0.5945778 (-1.438 %), -u = 0.2446483 (-3.841 %), 14.9 s
  plate_fsdt_tl          w = 0.5926049 (-1.765 %), -u = 0.2424832 (-4.692 %), 10.8 s
  plate_tsdt_gl          w = 0.7277065 (+20.630 %), -u = 0.5010012 (+96.919 %), 9.1 s

3. thick strip L = 1, b = h = 0.2 (L/h = 5), nu = 0, 14 x 4 terms, vs Reissner beam with kGA from the linear tip deflection of the model
  alpha = 1 plate_tsdt_tlA_proto   kGA/(GA) = 0.8353, tip rotation 0.461, w = 0.308092 (ref 0.308788, -0.225 %), -u = 0.058243 (ref 0.057572, +1.166 %), 2.7 s
  alpha = 1 plate_tsdt_tlB_proto   kGA/(GA) = 0.8353, tip rotation 0.461, w = 0.308752 (ref 0.308788, -0.012 %), -u = 0.058737 (ref 0.057572, +2.024 %), 3.5 s
  alpha = 1 plate_fsdt_tl          kGA/(GA) = 0.8255, tip rotation 0.461, w = 0.308346 (ref 0.308864, -0.168 %), -u = 0.058350 (ref 0.057595, +1.311 %), 2.5 s
  alpha = 1 plate_tsdt_gl          kGA/(GA) = 0.8353, tip rotation 0.461, w = 0.341314 (ref 0.308788, +10.533 %), -u = 0.073244 (ref 0.057572, +27.222 %), 1.5 s
  alpha = 3 plate_tsdt_tlA_proto   kGA/(GA) = 0.8353, tip rotation 0.981, w = 0.597807 (ref 0.619057, -3.433 %), -u = 0.237442 (ref 0.257802, -7.898 %), 6.5 s
  alpha = 3 plate_tsdt_tlB_proto   kGA/(GA) = 0.8353, tip rotation 0.981, w = 0.614051 (ref 0.619057, -0.809 %), -u = 0.257283 (ref 0.257802, -0.201 %), 7.6 s
  alpha = 3 plate_fsdt_tl          kGA/(GA) = 0.8255, tip rotation 0.981, w = 0.610289 (ref 0.619176, -1.435 %), -u = 0.252317 (ref 0.257871, -2.154 %), 5.8 s
  alpha = 3 plate_tsdt_gl          kGA/(GA) = 0.8353, tip rotation 0.981, w = 0.744870 (ref 0.619057, +20.323 %), -u = 0.530059 (ref 0.257802, +105.607 %), 5.3 s

4. thick strip b = h = 0.2, tip force along (0, 1, 1)/sqrt(2), alpha = 2 (out-of-plane EI), 14 x 6 terms, model against model
  plate_tsdt_tlA_proto   (u, v, w)/L = (-0.073273, 0.276775, 0.216200), 15.0 s
  plate_tsdt_tlB_proto   (u, v, w)/L = (-0.169596, 0.361435, 0.358034), 18.8 s
  plate_fsdt_tl          (u, v, w)/L = (-0.169034, 0.361669, 0.357428), 13.3 s
  plate_tsdt_gl          (u, v, w)/L = (-0.219686, 0.364354, 0.427783), 7.2 s
  A - B relative to |B|: 3.566e-01

5. simply supported square plate a = 1, h = 0.2, nu = 0.3, uniform pressure, 12 x 12 terms, centre w/h
  plate_tsdt_tlA_proto   p = 2.5e+08: 0.13386, p = 5e+08: 0.26213, p = 1e+09: 0.48921, p = 1.5e+09: 0.67451, 32.7 s
  plate_tsdt_tlB_proto   p = 2.5e+08: 0.13387, p = 5e+08: 0.26221, p = 1e+09: 0.48971, p = 1.5e+09: 0.67582, 44.0 s
  plate_fsdt_tl          p = 2.5e+08: 0.13416, p = 5e+08: 0.26278, p = 1e+09: 0.49070, p = 1.5e+09: 0.67686, 30.3 s
  plate_tsdt_gl          p = 2.5e+08: 0.13401, p = 5e+08: 0.26322, p = 1e+09: 0.49599, p = 1.5e+09: 0.69187, 16.8 s
  plate_tsdt_tlA_proto   relative to B: -0.008 %, -0.029 %, -0.102 %, -0.194 %
  plate_tsdt_tlB_proto   relative to B: +0.000 %, +0.000 %, +0.000 %, +0.000 %
  plate_fsdt_tl          relative to B: +0.216 %, +0.218 %, +0.203 %, +0.153 %
  plate_tsdt_gl          relative to B: +0.102 %, +0.385 %, +1.283 %, +2.374 %

6. cost: source and compiled sizes; time of calc_kC(NLgeom) + calc_kG + calc_fint, plate 12 x 12 terms, best of 3
  plate_tsdt_tlA_proto   pyx 2086 lines 97 kB, compiled 558 kB, kC 0.658 s, kG 0.211 s, fint 0.047 s
  plate_tsdt_tlB_proto   pyx 2464 lines 136 kB, compiled 622 kB, kC 0.884 s, kG 0.360 s, fint 0.054 s
  plate_fsdt_tl          pyx 2392 lines 111 kB, compiled 614 kB, kC 0.560 s, kG 0.190 s, fint 0.040 s
  plate_tsdt_gl          pyx 2399 lines 107 kB, compiled 610 kB, kC 0.317 s, kG 0.065 s, fint 0.022 s
```
