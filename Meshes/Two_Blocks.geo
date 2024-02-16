// Gmsh project created on Mon Feb 12 16:24:54 2024
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Transfinite Curve {3, 2, 1, 4} = 1 Using Progression 1;
//+
Transfinite Surface {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {1}; Recombine;
}
//+
Point(9) = {0, 1.1, 0, 1.0};
//+
Point(10) = {1, 1.1, 0, 1.0};
//+
Point(11) = {1, 2, 0, 1.0};
//+
Point(12) = {0, 2, 0, 1.0};
//+
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};
//+
Transfinite Curve {16, 15, 14, 13} = 1 Using Progression 1;
//+
Curve Loop(7) = {16, 13, 14, 15};
//+
Plane Surface(7) = {7};
//+
Transfinite Surface {7};
//+
Extrude {0, 0, 1} {
  Surface{7}; Layers {1}; Recombine;
}
//+
Physical Volume(25) = {1};
//+
Physical Volume(26) = {2};
