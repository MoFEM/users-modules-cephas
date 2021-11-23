# Photon diffusion

```bash
../../tools/split_sideset -my_file \
~/tmp/cross.cub \
-side_sets 1 \
-squash_bit_levels \
-output_vtk \
-split_corner_edges
```

```bash
../../tools/mofem_part \
-my_file out.h5m \
-output_file ab.h5m \
-my_nparts 4 -dim 3 -adj_dim 0
```

```bash
mpirun -np 4 ./inital_diffusion \
-file_name ab.h5m \
-order 3 \
-ksp_type gmres \
-ksp_monitor \ 
-my_max_post_proc_ref_level 2
```

```bash
mpirun -np 4 \
./photon_diffusion \
-file_name ab.h5m \
-order 4 \
-ts_max_time 4 \
-ts_dt 0.1 \
-my_max_post_proc_ref_level 2
```
