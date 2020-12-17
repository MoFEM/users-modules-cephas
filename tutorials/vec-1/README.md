Resources
=========

Test taken from:
https://cofea.readthedocs.io/en/latest/benchmarks/000-tuning-fork/index.html
https://github.com/spolanski/CoFEA/blob/master/docs/benchmarks/000-tuning-fork/index.md

Data
====

- Young's modulus, E=207 [GPa]
- Poissons ratio, nu = 0.33
- Material density, \rho = 7829 [kg/m^{3}]


Example command line
====================

Partitioning mesh
```
mofem_part \
-my_file fork-2-0.h5m \
-output_file fork-2-0_4parts.h5m \
-nparts 3 -dim 4 -adj_dim 1
```

Running code

```
mpirun -np 4 ./eigen_elastic_3d \
-file_name fork-2-0_4parts.h5m -eps_monitor -order 3 \
-eps_pos_gen_non_hermitian -eps_ncv 200 \
-eps_tol 1e-3 -eps_nev 10
```