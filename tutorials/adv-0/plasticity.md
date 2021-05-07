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

import os
os.system('/usr/bin/Xvfb :%d -screen 0 1024x768x24 &' % os.getuid())
os.environ['DISPLAY'] = ':%d' % os.getuid()
os.environ['PYVISTA_USE_IPYVTK'] = 'true'
print(os.environ['DISPLAY'])

# log file
log_file='log'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['figure.figsize'] = [15, 10]

```

```python
#mesh_file='plate_with_hole_displacement_control.h5m'
mesh_file='plate_with_hole_force_control.h5m'
!mofem_part -file_name {mesh_file} -my_nparts 1 -dim 2 -output_file one.h5m
```

```python
# Material parameters
young_modulus=200000
poisson_ratio=0.3
hardening=2
yield_stress=200

number_of_steps=75
nb_elastic_steps=30
max_von_misses_fisrt_elastic_step = 4.647031123

# Load step and tolarence
elastic_step=False

if elastic_step:
    load_step=1
    final_load=1*load_step
    yield_stress=1e32 # Make yield stress very big to have elastic step
else:
    load_step=(yield_stress/max_von_misses_fisrt_elastic_step)/nb_elastic_steps
    final_load=number_of_steps*load_step

realative_tolerance=1e-12
absolute_tolerance=1e-12

!rm -f out_*
!./plastic_2d \
-file_name one.h5m \
-ts_dt {load_step} \
-ts_max_time {final_load} \
-snes_atol {absolute_tolerance} \
-snes_rtol {realative_tolerance} \
-large_strains 0 \
-scale 1 \
-order 2 \
-young_modulus {young_modulus} \
-poisson_ratio {poisson_ratio} \
-hardening {hardening} \
-hardening_viscous 0 \
-yield_stress {yield_stress} \
-log_quiet 2>&1 | tee log

# Plot convergence
!grep SNES {log_file} > snes
newton_data=pd.read_fwf('snes', header=None)
newton_data=newton_data.rename(columns={0: "it", 4: "res", 6: "equlibrium", 8: 'constrain', 10: 'flow'})
newton_data=newton_data.drop([1, 2, 3, 5, 7, 9, 11], axis=1)

newton_data

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
for i in sorted_list_of_files:
    !mbconvert {i} {i[:-3]}vtk

import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
import re, os

my_cmap = plt.cm.get_cmap("rainbow", 16)

scale_displacements=0.5

list_of_files=!ls -c1 out*.vtk
list_of_files=sorted(list_of_files, key=extract_numner)
mesh = pv.read(list_of_files[len(list_of_files)-1])
print(mesh)

mesh=mesh.warp_by_vector('U',factor=scale_displacements)
print('Max von misses stress ', max(mesh["PLASTIC_SURFACE"]))

mesh.plot(
        show_grid=True,
        show_edges=True,
        cpos="xy",
        scalars="PLASTIC_SURFACE", 
        smooth_shading=True, 
        cmap=my_cmap
    )
```

```python
# Plot solution to PNG files

pv.set_plot_theme("document")
list_of_files=!ls -c1 out*.vtk
sorted_list_of_files = sorted(list_of_files, key=lambda f: int(re.sub('\D', '', f)))
for f in sorted_list_of_files:
    
    step_num = int(''.join(filter(str.isdigit, f)))        
    
    mesh = pv.read(f)
    mesh=mesh.warp_by_vector('U',factor=scale_displacements)

    print("Write file", os.path.splitext(f)[0]+'.png')
    # Take a screen shot
    mesh.plot(
        window_size=[1024, 768],
        off_screen=True,
        notebook=False,
        screenshot=os.path.splitext(f)[0]+'.png',
        show_grid=False,
        show_edges=True,
        text='step: %d' % (step_num),
        cpos="xy",
        scalars="PLASTIC_MULTIPLIER", 
        smooth_shading=True, 
        cmap=my_cmap
    )
```

```python
# Render animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.image as mpimg
import re, os
from IPython.display import HTML
plt.rcParams['animation.html'] = 'jshtml'
# plt.rcParams['animation.embed_limit'] = 20000000
plt.rcParams['figure.dpi'] = 150
list_of_pngs =!ls -c1 out*.png
sorted_list_of_pngs= sorted(list_of_pngs, key=lambda f: int(re.sub('\D', '', f)))
snapshots = []
for p in sorted_list_of_pngs:
    img = mpimg.imread(p)
    snapshots.append(img)
fps = 1
frame_num = len(sorted_list_of_pngs)
nSeconds = frame_num / fps
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,6) )
a = snapshots[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )
    im.set_array(snapshots[i])
    return [im]
anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = frame_num,
                               interval = 1000 / fps, # in ms
                               )
plt.axis('off')
plt.close()
print('Rendering animation ', end ='' )
HTML(anim.to_jshtml())
anim
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
