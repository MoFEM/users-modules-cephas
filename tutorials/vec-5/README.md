```
../..//tools/mofem_part \\
-my_file eye.cub \\
-output_file ab.h5m \\
-my_nparts 6 -dim 2 -adj_dim 1
```

```
mpirun -np 6 ./free_surface \
-ts_dt 2e-3 -ts_max_time 200 \
-file_name ab.h5m \
-log_sl inform \
-snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 \
-ts_adapt_type none
```