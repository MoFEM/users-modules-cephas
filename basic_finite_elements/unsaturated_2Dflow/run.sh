#!/bin/bash
  
../../tools/add_meshsets -my_file paper_network.msh -meshsets_config bc.cfg

cp out.h5m paper_network_pre.h5m

../../tools/mofem_part -my_file paper_network_pre.h5m -output_file paper_network_final.h5m -my_nparts 1 -dim 2 -adj_dim 1

time mpirun -np 1 ./unsatu2dFlow_prob 2>&1 | tee log

../nonlinear_elasticity/do_vtk.sh out_*.h5m