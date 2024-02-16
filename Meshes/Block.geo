// Gmsh project created on Sat Feb 10 21:03:19 2024
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Transfinite Curve {4, 3, 2, 1} = 2 Using Progression 1;
//+
Transfinite Surface {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {1}; Recombine;
}
//+
Physical Volume(13) = {1};
