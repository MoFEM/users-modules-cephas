# Readme for eiegn problem

## Resources

Test setup, data, and meshes from:
https://cofea.readthedocs.io/en/latest/benchmarks/000-tuning-fork/index.html
https://github.com/spolanski/CoFEA/blob/master/docs/benchmarks/000-tuning-fork/index.md

## Data

- Young's modulus, E=207 [GPa]
- Poissons ratio, nu = 0.33
- Material density, \rho = 7829 [kg/m^{3}]


## Example command line

Convert from med format
```
read_med \
-med_file TET-2-0.med \
-output_file fork-2-0.h5m 
```

Partitioning mesh
```
mofem_part \
-my_file fork-2-0.h5m \
-output_file fork-2-0_4parts.h5m \
-nparts 4 -dim 3 -adj_dim 1
```

## Running code

```
mpirun -np 4 ./eigen_elastic_3d \
-file_name fork-2-0_4parts.h5m -eps_monitor -order 2 \
-eps_pos_gen_non_hermitian -eps_ncv 200 \
-eps_tol 1e-3 -eps_nev 1
```

## Results for tuning fork

- In analysis linear geometry was used.
- Frequency was calculated form  eq. f = sqrt(omgea2)/(2 * pi), where omega2 is
eiegn value for genrlaied problem: Ku  = omaga2*Mu, where K is stiffens matrix, 
and M is mass matrix.
- SLEPC type KrylovShur solver is used.

### Mesh TET-2-0.med

| order | DOFs    | Mode 1   | Mode 2    | Mode 3    | Mode 4    | Mode 5    |
--------|---------|----------|-----------|-----------|-----------|-----------|
| 1     | 2361    | 564.4 Hz | 1014.4 Hz | 2113.8 Hz | 2766.2 Hz | 3532.0 Hz |
| 2     | 13131   | 444.1 Hz | 680.4 Hz  | 1693.3 Hz | 1829.9 Hz | 2796.7 Hz |
| 3     | 38646   | 442.9 Hz | 676.9 Hz  | 1691.8 Hz | 1826.4 Hz | 2790.8 Hz |

### Mesh TET-1-0.med

| order | DOFs    | Mode 1   | Mode 2    | Mode 3    | Mode 4    | Mode 5    |
--------|---------|----------|-----------|-----------|-----------|-----------|
| 1     | 9876    | 490.9 Hz | 782.1 Hz  | 1880.4 Hz | 2090.3 Hz | 3101.6 Hz |
| 2     | 63063   | 441.3 Hz | 675.2 Hz  | 1690.3 Hz | 1826.4 Hz | 2783.1 Hz |
| 3     | 196323  | 441.0 Hz | 674.4 Hz  | 1690.3 Hz | 1825.7 Hz | 2781.3 Hz |

### Mesh TET-0-5.med

| order | DOFs    | Mode 1   | Mode 2    | Mode 3    | Mode 4    | Mode 5    |
--------|---------|----------|-----------|-----------|-----------|-----------|
| 1     | 56556   | 455.6 Hz | 703.1 Hz  | 1746.3 Hz | 1893.2 Hz | 1893.2 Hz |
| 2     | 396498  | 440.3 Hz | 673.9 Hz  | 1689.5 Hz | 1825.7 Hz | 2778.1 Hz |


![fork](http://mofem.eng.gla.ac.uk/mofem/html/fork.gif "Tuning fork for oder 3 and TET-1-0")