// Gmsh project created on Fri Dec 27 13:27:57 2019
//+
lx = 0.02;
ly = 0.005;
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, lx, ly, 0};

//+
Rotate {{0, 0, 1}, {lx, ly/2, 0}, Pi/3} {
  Duplicata { Surface{1}; }
}

Rotate {{0, 0, 1}, {lx, ly/2, 0}, 2*Pi/3} {
  Duplicata { Surface{1}; }
}

Rotate {{0, 0, 1}, {lx, ly/2, 0}, Pi} {
  Duplicata { Surface{1}; }
}

Rotate {{0, 0, 1}, {lx, ly/2, 0}, 3*Pi/2} {
  Duplicata { Surface{1}; }
}

BooleanUnion { Surface{1}; } { Surface{2}; }

Recursive Delete { Surface{1}; Surface{2}; }

BooleanUnion { Surface{6}; } { Surface{3}; }

Recursive Delete {  Surface{6}; Surface{3}; }

BooleanUnion { Surface{7}; } { Surface{4}; }

Recursive Delete {  Surface{7}; Surface{4}; }

BooleanUnion { Surface{8}; } { Surface{5}; }

Recursive Delete {  Surface{8}; Surface{5}; }

Physical Curve("ESSENTIAL") = {8, 12, 30};

Physical Surface("INITIAL") = {9};

Physical Surface("REGION1") = {9};


//+
Show "*";
//+
Show "*";
