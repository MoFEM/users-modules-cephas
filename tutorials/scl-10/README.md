This command line is used to run a steady-state heat analysis with the conduction as input "-D"

./thermal_2d -file_name mesh2d_test2.cub -order 2 -D 10

Use the following command to convert the output file into a format compatible with Paraview.

mbconvert out_result.h5m out_result_D10.vtk
