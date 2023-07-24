# Running code

~~~~
../../tools/mofem_part \
-my_file elasto_thermal_mesh.cub -output_file mesh.h5m -my_nparts 4 -dim 2 -adj_dim 1
~~~~

~~~~
mpirun -np 4 ./thermo_elastic_2d \
-file_name mesh.h5m \
-ts_max_time 1 -ts_adapt_type none \
-ts_dt 0.25 -ksp_monitor -order 6 -max_post_proc_ref_level 2
~~~~