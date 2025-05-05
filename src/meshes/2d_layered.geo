// 2D domain with horizontal strips that can have different material parameters

Mesh.SaveAll = 1;
Mesh.Smoothing = 100;

lc = Pi / 32.0;
width = 2*Pi;
full_height = Pi;
layer_count = 4;
layer_height = full_height / layer_count;

// points and horizontal lines
For i In {0 : layer_count}
  pl = newp; Point(pl) = {0, i * layer_height, 0, lc};
  pr = newp; Point(pr) = {width, i * layer_height, 0, lc};
  points_l[i] = pl;
  points_r[i] = pr;
  l = newl; Line(l) = {pl, pr};
  lines_hor[i] = l;
EndFor

// left and right walls
For i In {0 : layer_count-1}
  ll = newl; Line(ll) = {points_l[i], points_l[i+1]};
  lr = newl; Line(lr) = {points_r[i], points_r[i+1]};
  lines_l[i] = ll;
  lines_r[i] = lr;
EndFor

// surfaces and physical groups
For i In {0 : layer_count-1}
  // I believe indices here must be 1 or greater, hence i+1
  Curve Loop(i+1) = {lines_hor[i], lines_r[i], -lines_hor[i+1], -lines_l[i]};
  Plane Surface(i+1) = {i+1};

  Physical Point(i+1) = {points_l[i], points_r[i], points_r[i+1], points_l[i+1]};
  Physical Curve(i+1) = {lines_hor[i], lines_r[i], lines_hor[i+1], lines_l[i]};
  Physical Surface(i+1) = {i+1};
EndFor

// each boundary also gets its own group
// for source terms and energy measurements
// (named physical groups not currently supported
// so we just give them arbitrary identifiable numbers)

// bottom
Physical Point(990) = {points_l[0], points_r[0]};
Physical Curve(990) = {lines_hor[0]};
// top
Physical Point(991) = {points_l[layer_count], points_r[layer_count]};
Physical Curve(991) = {lines_hor[layer_count - 1]};
// right
Physical Point(992) = {points_r[]};
Physical Curve(992) = {lines_r[]};
// left
Physical Point(993) = {points_l[]};
Physical Curve(993) = {lines_l[]};

