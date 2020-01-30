#!/bin/bash
  
if  [[ $1 = "-my_file" ]]; then
    echo "Running with input file $2 ..."
    ../../tools/add_meshsets -my_file $2 -meshsets_config bc.cfg 
else
  echo "Give the input mesh file name ..."  
  exit
fi

cp out.h5m paper_network_pre.h5m

../../tools/mofem_part -my_file paper_network_pre.h5m -output_file paper_network_final.h5m -my_nparts 1 -dim 2 -adj_dim 1

if [[ -z "$3" ]]; then
    time mpirun -np 1 ./unsatu2dFlow_prob -file_name paper_network_final.h5m 2>&1 | tee log
elif [[ $3 = "-saturated_conductivity" ]]; then
    time mpirun -np 1 ./unsatu2dFlow_prob -file_name paper_network_final.h5m -saturated_conductivity $4 2>&1 | tee log
fi

../nonlinear_elasticity/do_vtk.sh out_*.h5m