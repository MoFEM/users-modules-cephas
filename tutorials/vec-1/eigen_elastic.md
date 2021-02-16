# Running problem

```python
import os
os.system('/usr/bin/Xvfb :99 -screen 0 1024x768x24 &')
os.environ['DISPLAY'] = ':99'
os.environ['PYVISTA_USE_IPYVTK'] = 'true'

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

user_name=!whoami
user_name=user_name[0]
print(user_name)

if user_name == 'root':
    home_dir = '/mofem_install'
else:
    home_dir=os.environ['HOME']

um_view_dir='%s/um_view' % home_dir
bin_dir=um_view_dir + '/bin'
working_dir=um_view_dir + '/tutorials/vec-1'

plt.rcParams['figure.figsize'] = [15, 10]
```

## Data

- Young's modulus, E [$GPa$]
- Poissons ratio, $\nu$ 
- Material density, $\rho$ [$kg/m^{3}$]

```python
YoungModulus=207
nu=0.3
rho=7829
```

## Preparing mesh

If you have mesh in native hd5 MOAB format, VTK, Cubit, gMesh, you can use such mesh directly. If you using mesh in MED format used by code-aster, you need to convert mesh into native MOAB hdf5 format.

```python
#!cd {working_dir} && {bin_dir}/read_med \
#-med_file TET-2-0.med \
#-output_file fork-2-0.h5m; 
!ls
```

# Partition mesh

If you would like run analysis on multiple-processors you need to partition mesh. You can do that using mofem tool to do that.

```python
# Set number of processors
NumberOfCores=4

mesh_name='fork-2-0.h5m'
part_mesh='part_mesh.h5m'

# Parition mesh
!cd {working_dir} && {bin_dir}/mofem_part \
-my_file {mesh_name} \
-output_file {part_mesh} \
-nparts {NumberOfCores} -dim 3 -adj_dim 1

# Convert mesh to VTK format
!cd {working_dir} && {bin_dir}/mbconvert {part_mesh} out_part.vtk

# Print paritions
mesh = pv.read(working_dir+'/'+'out_part.vtk')
my_cmap = plt.cm.get_cmap("jet", 12)


mesh.plot(
    show_grid=True,
    show_edges=True,
    scalars="PARALLEL_PARTITION", 
    smooth_shading=False, 
    cmap=my_cmap)

```

# Calculating eigen values

```python
# Approximation order
order=1

# Tolerance
Tol=1e-3

# Number of eigen values to calculate
numberOfValuesToCalulate=5

# Clean previous solution
!cd {working_dir} && rm -vf out_eig_*

# Log file
log_file='log'

# Running code
!cd {working_dir} && \
{bin_dir}/mpirun --allow-run-as-root -np {NumberOfCores} ./eigen_elastic_3d \
-file_name {part_mesh} -eps_monitor -order {order} \
-eps_pos_gen_non_hermitian -eps_ncv 400 \
-eps_tol {Tol} -eps_nev {numberOfValuesToCalulate} -log_no_color 2>&1 | tee {log_file}
```

# Post process

```python
# Get frequencies 
frequency_log='frequency_log'
!cd {working_dir} && grep "frequency" {log_file} > {frequency_log} 
frequencies_data=pd.read_csv(working_dir+"/"+frequency_log,sep='\s+',header=None)
frequencies=frequencies_data[14].to_numpy()
print(frequencies)

for f in zip(range(1,numberOfValuesToCalulate+1),frequencies):
    print('Mode ',f[0],' frequency ',f[1],' Hz')
    
plt.barh(range(1,numberOfValuesToCalulate+1), frequencies, align='center')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Vibration mode')
```

```python
# Convert files to VTK
out_files=!ls {working_dir}/out_eig_*.h5m
for f in out_files:
    !cd {working_dir} && {bin_dir}/mbconvert {f} {f}'.vtk'

for f in range(0,numberOfValuesToCalulate):
    plot_mode=f
    plot_file=('%s/out_eig_%d.h5m.vtk') % (working_dir,plot_mode)
    print(plot_file)

    mesh = pv.read(plot_file)
    my_cmap = plt.cm.get_cmap("jet", 24)

    # Take a screen shot
    max_u = 0
    for u in mesh.point_arrays['U']:
        max_u=max(max_u, u[0]**2+u[1]**2+u[2]**2)

    max_u=np.sqrt(max_u)
    print('Max displacement ',max_u)

    mesh=mesh.warp_by_vector('U',factor=10/max_u)
    mesh.plot(
        screenshot=('%s/screenshot_%d.png') % (working_dir,plot_mode),
        show_grid=False,
        show_edges=False,
        scalars="U", 
        smooth_shading=False, 
        cmap=my_cmap)        
```

```python
# # Plot screan shots

# for f in range(0,numberOfValuesToCalulate):
#     screeen_shot=('%s/screenshot_%d.png') % (working_dir,f)
#     img = mpimg.imread(screeen_shot)
#     imgplot = plt.imshow(img)
#     plt.show()
    
```

# Note

Example mesh and data were taken form [CoFEA](https://cofea.readthedocs.io/en/latest/benchmarks/000-tuning-fork/index.html)
