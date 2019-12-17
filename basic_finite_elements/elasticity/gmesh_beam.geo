// Gmsh project created on Thu Dec 20 09:20:14 2018
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 10, 1, 1};

Physical Volume("ElasticBlock") = {1};
Physical Surface("DispBC") = {1};
Physical Surface("PressureBC") = {4};
