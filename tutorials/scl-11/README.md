This command line is used to run a steady-state heat analysis with the conduction as input "-D"

./thermal*2d -file_name mesh2d_test_flux*.cub -order 4 -D 1. -bc_temp1 0. -bc_flux1 0.

Use the following command to convert the output file into a format compatible with Paraview.

mbconvert out_result.h5m out_result_D10.vtk

Use the following commands for thermo-elastic problem

./thermal_elastic_2d -file_name beam.cub -order 4 -D 5. -bc_flux1 20. -bc_flux2 0. -source 6
