
# Use read_med to create *.h5m file from *.med file (exported from Gmsh *.geo file)

read_med -med_file square.med -meshsets_config  square.config -output_file square.h5m
read_med -med_file Lshape.med -meshsets_config  Lshape.config -output_file Lshape.h5m

*Note:* `read_med` is located at `/mofem_install/mofem-cephas/mofem/users_modules/um-build-RelWithDebInfo-abcd1234/tools/`

# Use mofem_part to partition the mesh if necessary

mofem_part -my_file square.h5m -output_file square_2parts.h5m -my_nparts 2 -dim 2 -adj_dim 1
mofem_part -my_file Lshape.h5m -output_file Lshape_2parts.h5m -my_nparts 2 -dim 2 -adj_dim 1
