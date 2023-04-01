---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
wd=!pwd
wd=wd[0]
print(wd)
```

```bash
../../tools/mofem_part -file_name mesh.cub -output_file mesh_out.h5m -nparts 4 -dim 2 -adj_dim 1
../../tools/uniform_mesh_refinement -file_name mesh_out.h5m -output_file um_out.h5m -dim 2 -nb_levels 3 -shift 0

```

```bash
export OMPI_MCA_btl_vader_single_copy_mechanism=none
mpirun -np 4 \
./level_set_2d \
-file_name um_out.h5m \
-ksp_monitor \
-max_post_proc_ref_level 0 \
-ts_dt 0.01 \
-ksp_monitor \
-ts_max_steps 30000
```

```bash
make clean
make -j32 
```

```python

```
