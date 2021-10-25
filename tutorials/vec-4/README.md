


```
./approx_sphere \
-file_name sphere.cub \
-snes_stol 0 \
-ksp_monitor \
-snes_linesearch_type l2 \
-ts_type beuler -ts_dt 1 \
-snes_rtol 1e-6 -order 4
```

```
 mofem_part \
 -my_file out_ho_mesh.h5m \
 -output_file ab.h5m \
 -my_nparts 6 \
 -dim 2 \
 -adj_dim 0
```

```
mpirun -np 6 ./shallow_wave \
-file_name ab.h5m \
-ts_dt 1e2 \
-ts_max_time 100000000000000 \
-ts_max_steps 5000 \
-snes_atol 1e-3 -snes_rtol 1e-10 \ -snes_linesearch_type bt \
-snes_rtol 1e-8 -snes_atol 1e-12 -order 4 \
-ts_type alpha -ts_alpha_radius 0.5
```




