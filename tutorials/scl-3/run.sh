#!/bin/bash
  
rm -rf out*

if [[ $1 = "-np" || $3 = "-order" ]]; then
  mofem_part -my_file mesh2d.cub -output_file mesh2d.h5m -my_nparts $2 -dim 2 -adj_dim 1
  # mofem_part -my_file mesh3d.cub -output_file mesh3d.h5m -my_nparts $2 -dim 3 -adj_dim 1

  # time mpirun -np $2 ./poisson_2d_homogeneous -file_name mesh2d.h5m -order $4
  #time mpirun -np $2 ./poisson_3d_homogeneous -file_name mesh3d.h5m -order $4
  #time mpirun -np $2 ./poisson_2d_nonhomogeneous -file_name mesh2d.h5m -order $4
  time mpirun -np $2 ./poisson_2d_lagrange_multiplier -file_name mesh2d.h5m -order $4

elif [[ $1 = "-order" ]]; then
  ../../tools/mofem_part -my_file mesh.cub -output_file mesh.h5m -my_nparts 1 -dim 2 -adj_dim 1

  time mpirun -np 1 ./poisson_2d_homogeneous -file_name mesh.h5m -order $2 2>&1
fi

# convert.py -np $2 out_level*
