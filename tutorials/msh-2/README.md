
# Use read_med to create *.h5m file from *.med file (exported from Gmsh *.geo file)

read_med -med_file cube.med -meshsets_config  cube.config -output_file cube.h5m

*Note:* `read_med` is located at `/mofem_install/mofem-cephas/mofem/users_modules/um-build-RelWithDebInfo-abcd1234/tools/`