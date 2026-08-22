function Matrix4() {
  let te = [];
  let i = 0;
  while (i < 16) {
    te[i] = (i % 5 === 0) ? 1.0 : 0.0;
    i = i + 1;
  }
  return { elements: te };
}

function getElem(m, idx) {
  let e = m.elements;
  return e[idx];
}

function setElem(m, idx, val) {
  let e = m.elements;
  e[idx] = val;
}

function multiplyMatrices(out, a, b) {
  let ae = a.elements;
  let be = b.elements;
  let te = out.elements;

  let a11 = ae[0 + 0], a12 = ae[0 + 4], a13 = ae[0 + 8], a14 = ae[0 + 12];
  let a21 = ae[0 + 1], a22 = ae[0 + 5], a23 = ae[0 + 9], a24 = ae[0 + 13];
  let a31 = ae[0 + 2], a32 = ae[0 + 6], a33 = ae[0 + 10], a34 = ae[0 + 14];
  let a41 = ae[0 + 3], a42 = ae[0 + 7], a43 = ae[0 + 11], a44 = ae[0 + 15];

  let b11 = be[0 + 0], b12 = be[0 + 4], b13 = be[0 + 8], b14 = be[0 + 12];
  let b21 = be[0 + 1], b22 = be[0 + 5], b23 = be[0 + 9], b24 = be[0 + 13];
  let b31 = be[0 + 2], b32 = be[0 + 6], b33 = be[0 + 10], b34 = be[0 + 14];
  let b41 = be[0 + 3], b42 = be[0 + 7], b43 = be[0 + 11], b44 = be[0 + 15];

  te[0 + 0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
  te[0 + 4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
  te[0 + 8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
  te[0 + 12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

  te[0 + 1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
  te[0 + 5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
  te[0 + 9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
  te[0 + 13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

  te[0 + 2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
  te[0 + 6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
  te[0 + 10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
  te[0 + 14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

  te[0 + 3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
  te[0 + 7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
  te[0 + 11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
  te[0 + 15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;

  return out;
}

function testMat4Mul(iterations) {
  let m1 = Matrix4();
  let m2 = Matrix4();
  let res = Matrix4();

  // Rotation matrix with cos(0.001) ~ 0.9999995, sin(0.001) ~ 0.0009999998
  setElem(m2, 0, 0.9999995);
  setElem(m2, 4, -0.0009999998);
  setElem(m2, 1, 0.0009999998);
  setElem(m2, 5, 0.9999995);
  setElem(m2, 12, 0.0001);
  setElem(m2, 13, 0.0002);

  let i = 0;
  while (i < iterations) {
    multiplyMatrices(res, m1, m2);
    setElem(m1, 0, getElem(res, 0));
    setElem(m1, 4, getElem(res, 4));
    setElem(m1, 1, getElem(res, 1));
    setElem(m1, 5, getElem(res, 5));
    setElem(m1, 12, getElem(res, 12) * 0.99999);
    setElem(m1, 13, getElem(res, 13) * 0.99999);
    i = i + 1;
  }

  let sum = 0.0;
  let k = 0;
  while (k < 16) {
    sum = sum + getElem(res, k);
    k = k + 1;
  }
  return sum;
}

function run() {
  let r = testMat4Mul(400000);
  print(r);
}

run();
