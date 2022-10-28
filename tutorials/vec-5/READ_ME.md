command line:

../../tools/mofem_part -my_file free_surface_mesh_extra_fine.cub -output_file mesh_free_surface.h5m -my_nparts 6 -dim 2 -adj_dim 1


make -j4 

mpirun -np 6 ./free_surface -ts_dt 1e-3 -ts_max_time 200 -file_name mesh_free_surface.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none


----------------------------------

//wetting angle test//

../../tools/mofem_part -my_file wetting_angle_cubit.cub -output_file wetting_angle_cubit.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 1e-3 -ts_max_time 200 -file_name wetting_angle_cubit.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor


----------------------------------

//simple test//

../../tools/mofem_part -my_file simple_wetting_angle_cubit.cub -output_file simple_wetting_angle_cubit.h5m -my_nparts 6 -dim 2 -adj_dim 1

*snes_fd version* 
mpirun -np 6 ./free_surface -ts_dt 2e-3 -ts_max_time 200 -file_name simple_wetting_angle_cubit.h5m -log_sl inform -snes_fd -ts_adapt_type none -snes_monitor

*normal test(not in cluding snes_fd)*
mpirun -np 6 ./free_surface -ts_dt 1e-3 -ts_max_time 200 -file_name simple_wetting_angle_cubit.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor


-----------------------------------

//finer mesh(half sample)

../../tools/mofem_part -my_file wetting_angle_half_cubit.cub -output_file wetting_angle_cubit_fine_80.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 1e-2 -ts_max_time 800 -file_name wetting_angle_cubit_fine_80.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor


-----------------------------------

//finer mesh(half sample-tall)

../../tools/mofem_part -my_file wetting_angle_half_cubit_tall.cub -output_file wetting_angle_cubit_fine_70_tall.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 1e-2 -ts_max_time 800 -file_name wetting_angle_cubit_fine_70_tall.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor


-----------------------------------

//finer mesh(half sample-tall)

../../tools/mofem_part -my_file wetting_angle_half_cubit_magnified.cub -output_file wetting_angle_cubit_fine_70_magnified.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 1e-2 -ts_max_time 800 -file_name wetting_angle_cubit_fine_70_magnified.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor

-----------------------------------

//super fine mesh(half sample)

../../tools/mofem_part -my_file super_fine_wetting_70.cub -output_file super_fine_wetting_70.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 5e-4 -ts_max_time 800 -file_name super_fine_wetting_70.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor



-----------------------------------

//super duper fine mesh 80 (half sample)

../../tools/mofem_part -my_file super_duper_fine_wetting_80.cub -output_file super_duper_fine_wetting_80.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 5e-4 -ts_max_time 800 -file_name super_duper_fine_wetting_80.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor



-----------------------------------

//super duper fine mesh 100 (half sample)

../../tools/mofem_part -my_file super_duper_fine_wetting_100.cub -output_file super_duper_fine_wetting_100.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 5e-4 -ts_max_time 800 -file_name super_duper_fine_wetting_100.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor



-----------------------------------

//super duper fine mesh 110 (half sample)

../../tools/mofem_part -my_file super_duper_fine_wetting_110.cub -output_file super_duper_fine_wetting_110.h5m -my_nparts 6 -dim 2 -adj_dim 1

mpirun -np 6 ./free_surface -ts_dt 5e-4 -ts_max_time 800 -file_name super_duper_fine_wetting_110.h5m -log_sl inform -snes_stol 0 -snes_atol 1e-10 -snes_rtol 1e-10 -ts_adapt_type none -snes_monitor

------------------------------------
../../tools/convert.py -np 4  out_step*h5m
zip out.zip out_step*.vtk