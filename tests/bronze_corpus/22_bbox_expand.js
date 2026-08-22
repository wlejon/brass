function Vector3(x, y, z) {
  return { x: x, y: y, z: z };
}

function Box3(min, max) {
  return { min: min, max: max };
}

function expandByPoint(box, point) {
  let min = box.min;
  let max = box.max;
  if (point.x < min.x) min.x = point.x;
  if (point.x > max.x) max.x = point.x;
  if (point.y < min.y) min.y = point.y;
  if (point.y > max.y) max.y = point.y;
  if (point.z < min.z) min.z = point.z;
  if (point.z > max.z) max.z = point.z;
  return box;
}

function unionBox(boxA, boxB) {
  expandByPoint(boxA, boxB.min);
  expandByPoint(boxA, boxB.max);
  return boxA;
}

function testBBoxExpand(iterations, pointCount) {
  let points = [];
  let p = 0;
  while (p < pointCount) {
    points[p] = Vector3(p * 0.15 - 25.0, p * 0.25 - 35.0, p * 0.35 - 45.0);
    p = p + 1;
  }

  let globalBox = Box3(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0));
  let iter = 0;
  while (iter < iterations) {
    let localBox = Box3(Vector3(1000.0, 1000.0, 1000.0), Vector3(-1000.0, -1000.0, -1000.0));
    let i = 0;
    while (i < pointCount) {
      expandByPoint(localBox, points[i]);
      i = i + 1;
    }
    unionBox(globalBox, localBox);
    let pt = points[iter % pointCount];
    pt.x = pt.x + 0.00001;
    iter = iter + 1;
  }

  let dx = globalBox.max.x - globalBox.min.x;
  let dy = globalBox.max.y - globalBox.min.y;
  let dz = globalBox.max.z - globalBox.min.z;
  return dx + dy + dz;
}

function run() {
  let r = testBBoxExpand(30000, 50);
  print(r);
}

run();
