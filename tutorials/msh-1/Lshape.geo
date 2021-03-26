//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {5, 0, 0, 1.0};
//+
Point(3) = {5, 5, 0, 1.0};
//+
Point(4) = {4, 5, 0, 1.0};
//+
Point(5) = {4, 1, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("BOUNDARY_CONDITION") = {6, 1, 2, 3, 4, 5};
//+
Physical Surface("main_surface", 7) = {1};
//+
Physical Curve("boundary", 8) = {5, 4, 3, 2, 1, 6};
