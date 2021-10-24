


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
-snes_rtol 1e-8 -snes_atol 1e-12 -order 4
```

```
-pc_type lu  
-pc_factor_mat_solver_type mumps  
#-ksp_monitor
#-ksp_converged_reason


-snes_atol 1e-8
-snes_rtol 1e-16
-snes_max_linear_solve_fail -1
-snes_converged_reason
-snes_linesearch_type basic
-snes_linesearch_monitor 
-snes_max_it 20
-snes_monitor

-ts_type basic
#-ts_theta_theta 0.5 
#-ts_theta_endpoint
#-ts_theta_theta 1
#-ts_theta_initial_guess_extrapolate 1

-ts_exact_final_time matchstep
-ts_max_time 10
-ts_dt 0.1
-ts_max_snes_failures -1
-ts_error_if_step_fails false
-ts_adapt_type none
-ts_monitor
-ts_adapt_monitor


-mat_mumps_icntl_14 800 
-mat_mumps_icntl_24 1 
-mat_mumps_icntl_13 1%
```




