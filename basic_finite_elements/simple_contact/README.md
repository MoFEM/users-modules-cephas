## Brief description 
Contact interaction between elastic solids having matching meshes in the contact interface.

## Definition of the contact interface(s)

Contacting solids should be merged (i.e. glued) along the apparent contact 
interface(s) in the input mesh. Each contact interface should be introduced in
the bulk of the mesh as a _BLOCKSET_ with a name starting with `INT_CONTACT`. An
arbitrary number of contact interfaces may be defined, each including an 
arbitrary number of meshed surfaces: 

![alt text](figures/contact_interface.png "Contact interface definition") *Contact interface definition* 

----

Below we consider several cases of definition of contact interfaces, outlining which ones are currently supported:  

**1.** Contact interface cutting through the whole body: **SUPPORTED**
![alt text](figures/contact_case_1.png "Case 1: Contact interface cutting through the whole body") *Case 1: Contact interface cutting through the whole body  (**SUPPORTED**)*

----

**2.** Contact interface cutting through a part of the body: **NOT SUPPORTED**
![alt text](figures/contact_case_2.png "Case 2: Contact interface cutting through a part of the body") *Case 2: Contact interface cutting through a part of the body (**NOT SUPPORTED**)*

----

**3.** Contact interface cutting through a part of the body and meeting another contact interface cutting through the whole body: **SUPPORTED**
![alt text](figures/contact_case_3.png "Case 3: Contact interface meeting another contact interface") *Case 3: Contact interface meeting another contact interface (**SUPPORTED**)*

----

**4.** Contact interface consisting of non-intersecting surfaces: **SUPPORTED**, given that the outlined conditions are satisfied, with the distance threshold considered as 3  elements in the bulk mesh  
![alt text](figures/contact_case_4.png "Case 4: Contact interface consisting of non-intersecting surfaces") *Case 4: Contact interface consisting of non-intersecting surfaces (**SUPPORTED**)*


<!-- The following rules for the definition of contact interfaces apply:
1. External edge of a contact interface may belong to the skin of the solid 
mesh (i.e. contact interface may cut through the whole solid).
2. External edge of a contact interface may belong to the interior of the 
solid mesh if and only if this edge also belongs to another contact interface,
which cuts through the whole solid (i.e. contact interfaces may intersect inside 
the mesh, but a contact interface cannot cut the solid mesh partially, e.g. 
similar to a crack notch).
3. If two surfaces do not have shared edges, but the distance between these 
surfaces is less than 3 elements in the bulk mesh, they must be included 
separately into two different contact interfaces. -->

## Creation of the _MED_ mesh file in _Salome_ 

***_NOTE:_*** In order to use the current contact algorithm the solids need to touch each other along the contact surface in the input mesh. Presented below is a rather general approach which permits to mesh the contacting solids separately, refine these meshes around the contact surface, and, finally, merge the meshes together.

### Geometry:

- Create the contacting solids separately 
- Use _Intersection_ to find common geometrical entities
- Create the contact surface using _Build -> Face_ from the intersection 
- _Partition_ solids with the intersection (one by one)
- Create all necessary groups for the partitioned solids:
    - Groups of volumes for solid blocks
    - Groups of edges for fixed BCs
    - Group of faces for load BCs
    - Group of faces for springs 
    - Group of faces for the contact interface (same name per contact interface for both solids)

***_NOTE:_*** See rules on definition of contact interfaces in the slides on theory. It is recommended to append the springs to the same surface where the load is applied.

### Mesh:

- Mesh each solid separately
- Create mesh groups from the geometry groups (for each solid separately)
- Mesh the contact surface (sufficiently fine)
- Create a group of faces from the meshed contact surface
- Create a _Submesh_ for each solid coming to contact:
    - _Geometry:_ contact group from the solid's geometry
    - _Algorithm:_ Import 1D-2D Elements from Another Mesh
    - _Hypothesis -> Source Faces:_ group of faces from the meshed contact surface
- Recompute meshes
- Build _Compound mesh_ of solids with node merging on
- Export the Compound mesh to the _MED_ file

## Preparation of the config file

- To see all block IDs in the _MED_ file:
```
read_med -med_file three_point_bending.med
```

- Check _BLOCKSET_ for the contact interface in the config file `three_point_bending.cfg`
```bash
# Contact interface
[block_2]               # Block ID in MED file 
id=2004                 # Block ID in the output *.h5m file
add=BLOCKSET            # Block type
name=INT_CONTACT        # Block name (starts exactly like this)
```
- Check _BLOCKSET_ for springs in the same config file
```bash
# Springs on the loading frame and on the brick slice
[block_8]               # Block ID in MED file 
id=2005                 # Block ID in the output *.h5m file
add=BLOCKSET            # Block type
name=SPRING_BC          # Block name (starts exactly like this)
user1=0                 # Spring stiffness in normal direction [MPa]
user2=1e-2              # Spring stiffness in tangential directions [MPa]
```
***_NOTE:_*** For the considered example the normal stiffness can be set initially to `0` (will be verified below), while the tangential one can be set to `1e-6` of the the Young's modulus of the solid to which these springs are attached. The calibration of this parameter will be discussed below.

## Generation of the _h5m_ file

- Generate `three_point_bending.h5m` file:
```
read_med -med_file three_point_bending.med \
-meshsets_config three_point_bending.cfg \
-output_file three_point_bending.h5m
```

- The correct definition of all blocks can be verified by generating `vtk` files for each one of them for visualisation in _Paraview_:
```
meshset_to_vtk -my_file three_point_bending.h5m
```

## Preparation of the param file 

Check following parameters in the param file `param_file.petsc`
```bash
-my_order 2         
-my_order_lambda 1          
-my_cn_value 1.e3                  
```

### Contact parameters:

Name | Description | Default value
--- | --- | ---
`my_order` | Approximation order of the field of spatial positions for the entire mesh | 1
`my_order_lambda` | Approximation order of the field of contact Lagrange multipliers | 1
`my_order_contact` | Approximation order of the field of spatial positions for the contact elements and a given number layers of tetrahedral elements adjacent to the contact interface | 1
`my_ho_levels_num` | Number of layers of tetrahedral elements adjacent to the contact interface with higher order of the field of spatial positions (if `my_order_contact` is greater than `my_order`) | 1 
`my_cn_value` | Augmentation parameter which affects the convergence and has minimal affect on the solution. Recommended initial value is the Young's modulus of contacting solids (or harmonic mean in case of different values). The optimal value can be found by repetitively increasing/decreasing the initial value by e.g. a factor of 10 | 1
`my_r_value` | Contact regularisation parameter which can lie between 1.0 and 1.1. Values greater than 1 can speed-up the convergence, but will also alter the stiffness of the contact interface, therefore it is not recommended to change this parameter | 1
`my_alm_flag` | Defines the choice of the algorithm: 0 (False) - Complementarity function approach, 1 (True) - Augmented Lagrangian method | 0  

## Running the contact simulation

```bash
mpirun -np 2 ./simple_contact -my_file examples/punch_top_only.cub \
-my_order 2 -my_cn_value 1e3
```

## Postprocessing

All the usual output files are created and can be postprocessed in the standard way. Furthermore, values of Lagrange multipliers (equivalent to contact pressure) and normal gap are output to files `out_contact_N.h5m`, where `N` is the number of the step. 

- Convert output `h5m` files to `vtk`
```bash
convert.py -np 2 out_contact*
```
***_NOTE:_*** The values of Lagrange multipliers and normal gap are computed at Gauss points and visualisation requires using the _Point Gaussian_ representation or alternatively the _Glyph_ filter

