
mofem_part -my_file mesh.cub -output_file mesh.h5m -my_nparts 2 -dim 2 -adj_dim 1

mpirun -np 2 ./minimal_surface_equation -file_name mesh.h5m -order 4