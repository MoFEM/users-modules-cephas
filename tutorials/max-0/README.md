Check before run
================

* Check where is your mesh file
* Check what version of openmpi (or other MPI library) are you using

If you are using spack and you would like to use mpirun and mbconvert 
tools please first run the following 
======================================================================
spack load moab

Example for running magnetostatic
=================================
mpirun -np 2 ./magnetostatic -my_file magnetic_coil_2parts.h5m \
-my_order 2 \
-my_max_post_proc_ref_level 1

Parameters set by default from param_file.petsc file
============================

-ksp_type fgmres 
-pc_type lu 
-pc_factor_mat_solver_type mumps 
-mat_mumps_icntl_13 1 
-ksp_monitor 
-mat_mumps_icntl_24 1 
-mat_mumps_icntl_13 1
