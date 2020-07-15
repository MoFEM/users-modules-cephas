#!/bin/bash
  
rm -rf out*

if [[ $1 = "-np" || $3 = "-order" ]]; then
        /Users/hoangnguyen/mofem_install/um/build_release/tools/mofem_part -my_file mesh.cub -output_file mesh.h5m -my_nparts $2 -dim 2 -adj_dim 1

	      time mpirun -np $2 ./wave_equation -file_name mesh.h5m -order_u $4 -order_v $4 2>&1
elif [[ $1 = "-order" ]]; then
        #cp /Users/hoangnguyen/mofem_install/mofem-cephas/mofem/users_modules/um_basics/poisson/mesh.cub /Users/hoangnguyen/mofem_install/um/build_release/um_basics/poisson
        
        /Users/hoangnguyen/mofem_install/um/build_release/tools/mofem_part -my_file mesh.cub -output_file mesh.h5m -my_nparts 1 -dim 2 -adj_dim 1

        # make -j1 -C /Users/hoangnguyen/mofem_install/um/build_release/um_basics/poisson

        time mpirun -np 1 /Users/hoangnguyen/mofem_install/um/build_debug/um_basics/poisson/simple_poisson_2d -file_name mesh.h5m -order $2 2>&1
fi


# convert.py -np 6 out_level*

# ./nonlinear_poisson_2d -file_name mesh.h5m -order 2 -snes_monitor -snes_converged_reason  -snes_type newtonls -snes_linesearch_type cp -snes_linesearch_monitor
