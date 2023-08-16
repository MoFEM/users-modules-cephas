#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
    
debug=True

def make_png(file, args): 
    my_cmap = plt.cm.get_cmap("turbo", 124)
    mesh = pv.read(file)
    
    if args.wrap_vector:
        mesh = mesh.warp_by_vector(args.wrap_vector, factor=1)
    
    p = pv.Plotter(notebook=False, off_screen=True)    
    if args.field:
        p.add_mesh(
            mesh, 
            scalars=args.field, 
            show_edges=False, 
            smooth_shading=False, cmap=my_cmap) 
    else:
       p.add_mesh(
            mesh, 
            show_edges=True, edge_color='white', color='white') 
    if args.d2:
        p.camera_position = args.d2
    p.camera.zoom(args.zoom)
    p.camera.roll += args.roll 
    p.camera.azimuth += args.azimuth
    image = p.screenshot('%spng' % file[:-3])
    
def is_not_vtk(files):
    return not files.endswith('vtk')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert multiple vtk files to png files using the pyvista.")
    parser.add_argument(
        "files", help="list of vtk files or a regexp mask", nargs='+')
    parser.add_argument('-d2', '--d2', dest='d2', default='')
    parser.add_argument('-f', '--field', dest='field', default='', type=str)
    parser.add_argument('-wv', '--wrap_vector', dest='wrap_vector', default='', type=str)
    parser.add_argument('--zoom', dest='zoom', default=1.2, type=float)
    parser.add_argument('--roll', dest='roll', default=0, type=float)
    parser.add_argument('--azimuth', dest='azimuth', default=0, type=float)
    args = parser.parse_args()
    
    if debug: 
        print(args)
    
    from pyvirtualdisplay import Display
    display = Display(backend="xvfb", visible=False, size=(800, 800))
    display.start()
    
    for f in args.files:
        make_png(f, args)

