


```
./approx_sphere \                      
-file_name ~/tmp/sphere.cub \
-snes_stol 0 \
-ksp_monitor \ 
-ts_type beuler -ts_dt 1 -ts_max_steps 1 \
-snes_rtol 1e-12 -snes_rtol 1e-12 -order 4 -pc_type lu
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
-file_name ab.h5m 
```




