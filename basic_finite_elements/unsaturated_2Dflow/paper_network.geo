// Gmsh project created on Mon Jan 27 21:51:41 2020
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.02, 0.005, 0};
Physical Surface("INITIAL", 1) = {1};
Physical Curve("ESSENTIAL", 2) = {4};

