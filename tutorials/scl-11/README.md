# Introduction
This directory contains the input and executable files for the tutorial [SCL-11: Poisson's equation with Discontinuous Galerkin Approach]

# Mesh generation

Meshes are generated by [Gmsh](https://gmsh.info). The tutorials for mesh
generation are available at
[MSH-1: Create a 2D mesh from Gmsh](http://mofem.eng.gla.ac.uk/mofem/html/basic_tutorials_mesh_generation_2d.html)
and [MSH-2: Create a 3D mesh from Gmsh](http://mofem.eng.gla.ac.uk/mofem/html/basic_tutorials_mesh_generation_3d.html).

The `*.h5m` meshes in this directory are copied from the directory `../msh-1/`
and `../msh-2/`
# Mesh partition

mofem_part -my_file mesh2d.cub -output_file mesh_1parts.h5m -my_nparts 1

*Note:* `mofem_part` is located at `/mofem_install/mofem-cephas/mofem/users_modules/um-build-RelWithDebInfo-abcd1234/tools/`
# Run the program
./poisson_2d_dis_galerkin -file_name mesh_1parts.h5m -order 4 -penalty 1e1 -phi -1 -nitsche 0 -log_sl inform
