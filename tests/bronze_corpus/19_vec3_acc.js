function Vector3(x, y, z) {
  return { x: x, y: y, z: z };
}

function addVectors(out, a, b) {
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  return out;
}

function multiplyScalar(v, s) {
  v.x = v.x * s;
  v.y = v.y * s;
  v.z = v.z * s;
  return v;
}

function testVec3Acc(iterations, arrSize) {
  let arr = [];
  let i = 0;
  while (i < arrSize) {
    arr[i] = Vector3(i * 0.1, i * 0.2, i * 0.3);
    i = i + 1;
  }
  let acc = Vector3(0.0, 0.0, 0.0);
  let iter = 0;
  while (iter < iterations) {
    let j = 0;
    while (j < arrSize) {
      addVectors(acc, acc, arr[j]);
      multiplyScalar(acc, 0.999999);
      j = j + 1;
    }
    iter = iter + 1;
  }
  return acc.x + acc.y + acc.z;
}

function run() {
  let res = testVec3Acc(20000, 100);
  print(res);
}

run();
