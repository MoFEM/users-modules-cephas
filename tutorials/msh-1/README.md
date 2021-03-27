
# Use read_med to create *.h5m file from *.med file (exported from Gmsh *.geo file)

read_med -med_file square.med -meshsets_config  square.config -output_file square.h5m
read_med -med_file Lshape.med -meshsets_config  Lshape.config -output_file Lshape.h5m

*Note:* `read_med` is located at `/mofem_install/mofem-cephas/mofem/users_modules/um-build-RelWithDebInfo-abcd1234/tools/`