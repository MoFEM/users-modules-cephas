
mofem_part -my_file mesh2d.cub -output_file mesh2d.h5m -my_nparts 2 -dim 2 -adj_dim 1

mpirun -np 2 ./poisson_2d_homogeneous -file_name mesh2d.h5m -order 4

mofem_part -my_file mesh3d.cub -output_file mesh3d.h5m -my_nparts 2 -dim 3 -adj_dim 1

mpirun -np 2 ./poisson_3d_homogeneous -file_name mesh3d.h5m -order 4