Use the following commands for thermo-elastic problem

./thermal_elastic_2d -file_name /Users/danielebarbera/mofem_install/users_modules/tutorials/scl-11/mesh_2D_test3.cub -order 4 -D 4. -bc_temp1 1000. -bc_temp2 1000. -set_body_force 0.

mbconvert out_result.h5m out_result_tm_Simple_test.vtk
