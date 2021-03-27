
# Use read_med to create *.h5m file from *.med file (exported from Gmsh *.geo file)

read_med -med_file cube.med -meshsets_config  cube.config -output_file cube.h5m

*Note:* `read_med` is located at `/mofem_install/mofem-cephas/mofem/users_modules/um-build-RelWithDebInfo-abcd1234/tools/`

# Use mofem_part to partition the mesh if necessary

mofem_part -my_file cube.h5m -output_file cube_2parts.h5m -my_nparts 2 -dim 3 -adj_dim 1
