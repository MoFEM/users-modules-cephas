Genarting base functions 
========================

Run on triangle
---------------

./plot_base_2d \
  -base ainsworth \
  -space h1 \
  -order 3 \
  -log_sl verbose  
  
convert.py -out_base_dof_*.h5m

Run on arbitrary mesh loaded by user
------------------------------------

./plot_base_2d \
  -load_file
  -file_name some_mesh_file.h5m \
  -base ainsworth \
  -space h1 \
  -order 3 \
  -log_sl verbose  


Options
=======

[-load_file] if provided mesh is load from file

[-file_name] mesh name

[-base ainsworth, ainsworth_labatto, demkowicz, bernstein]

[-space h1, l2, hcurl, hdiv]

[-oder o] where is approximation order

Note
====

Note tested for 3d. Not tested for hdiv in 3d.

