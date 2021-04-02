// Gmsh project created on Tue Mar 23 14:15:55 2021
SetFactory("OpenCASCADE");
//+
Box(1) = {-0.4, -2, -0.5, 1, 1, 1};
//+
Physical Volume("volume", 13) = {1};
//+
Physical Surface("top_surface", 14) = {6};
//+
Physical Surface("bottom_surface", 15) = {5};
//+
Physical Curve("bottom_edges", 16) = {9, 4, 11, 8};
//+
Physical Curve("top_edges", 17) = {10, 2, 12, 6};
