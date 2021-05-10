Command line to execute the beam problem

./thermal_2d -file_name beam.cub -order 4 -D 5. -bc_flux1 20. -bc_flux2 0. -source 6

Command line to convert the output in the desired vtk file

mbconvert out_result.h5m out_result_beam.vtk
