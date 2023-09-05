```
../../tools/mofem_part -my_file eye.cub -output_file out.h5m -my_nparts 6 -dim 2 -adj_dim 0 -my_file eye.cub -output_file out.h5m -my_nparts 12 -dim 2 -adj_dim 0
```

```
 ../../tools/uniform_mesh_refinement -file_name out.h5m -output_file um_out.h5m -dim 2 -nb_levels 2 -shift 0
 ```

```
mpirun --allow-run-as-root -np 12 ./free_surface -ts_dt 1e-4 -ts_max_time 200 -file_name um_out.h5m -log_sl inform -ksp_monitor -nb_levels 2 -refine_overlap 2 -coords cylindrical -h 0.005
```