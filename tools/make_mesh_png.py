#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
    
debug=True

def make_png(file, d2, field): 
    my_cmap = plt.cm.get_cmap("turbo", 124)
    mesh = pv.read(file)
    p = pv.Plotter(notebook=False, off_screen=True)
    if field:
        p.add_mesh(
            mesh, 
            scalars=field, 
            show_edges=False, 
            smooth_shading=False, cmap=my_cmap) 
    else:
       p.add_mesh(
            mesh, 
            show_edges=True, edge_color='white', color='white') 
    if d2:
        p.camera_position = "xy"
    p.camera.zoom(1.2)
    image = p.screenshot('%spng' %file[:-3])
    
def is_not_vtk(files):
    return not files.endswith('vtk')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert multiple h5m files to vtk files using the mbconvert tool (part of the MOAB library). The input files can be distributed between multiple processes to speed-up the conversion.")
    parser.add_argument(
        "files", help="list of vtk files or a regexp mask", nargs='+')
    parser.add_argument('-d2', '--d2', dest='d2', action='store_true')
    parser.add_argument('-f', '--field', dest='field', type=str)
    args = parser.parse_args()
    
    if debug: 
        print(args)
    
    from pyvirtualdisplay import Display
    display = Display(backend="xvfb", visible=False, size=(800, 800))
    display.start()
    
    for f in args.files:
        if 'field' in args:
            make_png(f, args.d2, args.field)
        else:
            make_png(f, args.d2, '')    

