Use the following commands for thermo-elastic problem

./thermal_elastic_2d -file_name beam.cub -order 4 -D 5. -bc_flux1 20. -source 6 -set_body_force 1.

mbconvert out_result.h5m out_result_D10.vtk
