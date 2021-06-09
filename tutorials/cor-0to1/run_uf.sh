#!/bin/bash
set -x

# exit when any command fails
set -e

# Get file name
if [ -z ${1+x} ]; then
  MESHFILE=1dTest.cub
else
  MESHFILE=$1
fi

# Get config file name
if [ -z ${2+x} ]; then
  CONFIGFILE=unsaturated.cfg
else
  CONFIGFILE=$2
fi

# Get numbet of processors
if [ -z ${3+x} ]; then
  NBPROCS=4
else
  NBPROCS=$3
fi

# Get step size
if [ -z ${4+x} ]; then
  DT=0.0001
else
  DT=$4
fi

# Get final time
if [ -z ${5+x} ]; then
  FT=1.0
else
  FT=$5
fi

# Get final time
if [ -z ${6+x} ]; then
  ORDER=0
else
  ORDER=$6
fi


# Partition mesh
../../tools/mofem_part -my_file $MESHFILE -meshsets_config $CONFIGFILE -my_nparts $NBPROCS

# Run code
rm -f out_*.h5m 
#make -j 4 unsaturated_transport 

mpirun --allow-run-as-root -np $NBPROCS \
./unsaturated_transport \
-my_file out.h5m  -configure $CONFIGFILE  \
-ts_monitor -ts_type beuler \
-ts_dt $DT -ts_final_time $FT \
-ts_monitor -ts_adapt_monitor \
-ts_adapt_type basic \
-ts_adapt_dt_max 0.005 \
-ts_adapt_dt_min 0.0001 \
-ts_rtol 0.01 -ts_atol 0.01 \
-ts_adapt_basic_safety 0.8 \
-ts_error_if_step_fails false \
-ts_theta_adapt false \
-ts_alpha_adapt false \
-ts_max_reject -1 \
-ts_max_snes_failures -1 \
-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps \
-snes_type newtonls \
-snes_linesearch_type l2 \
-snes_linesearch_minlambda 1e-3 -snes_linesearch_damping 1. -snes_linesearch_max_it 3 \
-snes_atol 1e-7 -snes_rtol 1e-4 -snes_stol 0 -snes_max_it 32 \
-snes_converged_reason \
-my_order $ORDER \
-how_often_output 5 \
-my_max_post_proc_ref_level 0 2>&1 | tee log 

# Will exit with status of last command.

exit $?