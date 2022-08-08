command line:

../../tools/mofem_part -my_file free_surface_mesh_extra_fine.cub -output_file mesh_free_surface.h5m -my_nparts 6 -dim 2 -adj_dim 1


make -j4 

mpirun -np 6 ./free_surface -ts_dt 2e-3 -ts_max_time 200 -file_name mesh_free_surface.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none


//wetting angle test//

../../tools/mofem_part -my_file wetting_angle_cubit.cub -output_file wetting_angle_cubit.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 2e-3 -ts_max_time 200 -file_name wetting_angle_cubit.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none


