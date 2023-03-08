Plasticity {#jup_plasticity}
===================== 

```python
import os

user_name=!whoami
id=!id -u {user_name}
user_name=user_name[0]
print(user_name)

if user_name == 'root':
    home_dir = '/mofem_install'
else:
    home_dir=os.environ['HOME'] 
    
wd=!pwd

print(home_dir)
print(wd)

from pyvirtualdisplay import Display
display = Display(backend="xvfb", visible=False, size=(800, 600))
display.start()

# log file
log_file='log'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['figure.figsize'] = [15, 10]

```

```python
nb_proc=4
mesh_file='plate_with_hole.h5m'
!mofem_part -file_name {mesh_file} -my_nparts {nb_proc} -dim 2 -output_file one.h5m
```

```python
# Material parameters
young_modulus=200000
poisson_ratio=0.3
hardening=2
yield_stress=200

number_of_steps=100
nb_elastic_steps=3
max_von_misses_fisrt_elastic_step = 33440.95912

# Load step and tolarence
elastic_step=False

if elastic_step:
    load_step=1
    final_load=1*load_step
    yield_stress=1e32 # Make yield stress very big to have elastic step
else:
    load_step=(yield_stress/max_von_misses_fisrt_elastic_step)/nb_elastic_steps
    final_load=number_of_steps*load_step

realative_tolerance=1e-9
absolute_tolerance=1e-9

!rm -f out_*
!export OMPI_MCA_btl_vader_single_copy_mechanism=none && \
mpirun --allow-run-as-root -np {nb_proc} ./plastic_2d \
-file_name one.h5m \
-ts_dt {load_step} \
-ts_max_time {final_load} \
-snes_atol {absolute_tolerance} \
-snes_rtol {realative_tolerance} \
-large_strains 0 \
-scale 1 \
-Qinf 0 \
-b_iso 0 \
-young_modulus {young_modulus} \
-poisson_ratio {poisson_ratio} \
-hardening {hardening} \
-hardening_viscous 0 \
-yield_stress {yield_stress} \
-zeta 1e-2 \
-log_quiet 2>&1 | tee log

```

```python
# Plot convergence
!grep Function {log_file} | sed 's/\[//g' | sed 's/,//g' > snes
newton_data=pd.read_fwf('snes', header=None)
newton_data=newton_data.rename(columns={0: "it", 3: "res", 4: "equlibrium", 5: 'flow', 6: 'constrain'})
newton_data=newton_data.drop([1, 2, 7], axis=1)
print(newton_data)

# newton_data

plt.plot(newton_data['res'].to_numpy(),'r^-')
plt.title('Neton method convergence')
plt.ylabel('absolute residial')
plt.xlabel('accumulated iterations')
plt.yscale('log')
plt.grid(True)
plt.show()
```

```python
# converting analysis files to format readable by post processors

!rm -f *.vtk

import re, os
list_of_files=!ls -c1 out_*.h5m
def extract_numner(s):
    return int(re.findall(r'\d+',s)[0])

size=len(list_of_files)
mod=int(size / 10)

list_to_process=[]
for f in list_of_files:
    n=extract_numner(f)
    if n == 0 or n == 1:
        list_to_process.append(f)
    elif n % mod == 0:
        list_to_process.append(f)
        
sorted_list_of_files = sorted(list_to_process, key=extract_numner)

out_to_vtk = !ls -c1 out_*h5m
last_file=out_to_vtk[0]
print(last_file)
!mbconvert {last_file} {last_file[:-3]}vtk
# for i in sorted_list_of_files:
#     !mbconvert {i} {i[:-3]}vtk

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import re, os

my_cmap = plt.cm.get_cmap("rainbow", 16)

scale_displacements=1

list_of_files=!ls -c1 out*.vtk
list_of_files=sorted(list_of_files, key=extract_numner)
mesh = pv.read(list_of_files[len(list_of_files)-1])
print(mesh)

mesh=mesh.warp_by_vector('U',factor=scale_displacements)
mesh=mesh.shrink(0.95)
print('Max von misses stress ', max(mesh["PLASTIC_SURFACE"]))

#field='U'
field='PLASTIC_STRAIN'
#field='PLASTIC_SURFACE'
#field='STRESS'
#field='STRAIN'
#field='FIRST_PIOLA'

p = pv.Plotter(notebook=True)
p.add_mesh(mesh, scalars=field,  show_edges=True, smooth_shading=False, cmap="rainbow")
p.camera_position = "xy"
p.show(jupyter_backend='ipygany')

```

```python
# Plot load disp-path
!grep reaction {log_file} > reaction
!grep Ux {log_file} > ux

data_reaction=pd.read_csv('reaction',sep='\s+',header=None)
data_ux=pd.read_csv('ux',sep='\s+',header=None)

data_ux=data_ux.rename(columns={2: "time", 4: "min", 6: "max"})
data_reaction=data_reaction.rename(columns={3: "reaction"})
table=pd.concat([data_reaction, data_ux.reindex(data_reaction.index)], axis=1)
table=table.drop([0, 1, 2, 3, 5], axis=1)

table['reaction']=-1*table['reaction'].to_numpy()

fig, ax = plt.subplots()
ax.plot(table['max'].to_numpy(), table['reaction'].to_numpy(), 'ro-')
ax.set(xlabel='displacement', ylabel='force',
       title='Load displacement path')
ax.grid(True)


```

```python

```
