function Quaternion(x, y, z, w) {
  return { x: x, y: y, z: z, w: w };
}

function sqrt(n) {
  if (n <= 0.0) return 0.0;
  let x = n;
  let i = 0;
  while (i < 12) {
    x = 0.5 * (x + n / x);
    i = i + 1;
  }
  return x;
}

function quatNormalize(q) {
  let l = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (l === 0.0) {
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
    return q;
  }
  let invLen = 1.0 / sqrt(l);
  q.x = q.x * invLen;
  q.y = q.y * invLen;
  q.z = q.z * invLen;
  q.w = q.w * invLen;
  return q;
}

function quatMultiply(out, a, b) {
  let qax = a.x, qay = a.y, qaz = a.z, qaw = a.w;
  let qbx = b.x, qby = b.y, qbz = b.z, qbw = b.w;

  out.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
  out.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
  out.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
  out.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;
  return out;
}

function testQuatNorm(iterations) {
  let q1 = Quaternion(0.1, 0.2, 0.3, 0.9);
  let q2 = Quaternion(0.01, 0.02, 0.03, 0.99);
  let rot = Quaternion(0.0, 0.0, 0.0, 1.0);

  let i = 0;
  while (i < iterations) {
    quatMultiply(rot, rot, q1);
    quatNormalize(rot);
    quatMultiply(rot, rot, q2);
    quatNormalize(rot);
    q1.x = q1.x + 0.0000001;
    i = i + 1;
  }
  return rot.x + rot.y + rot.z + rot.w;
}

function run() {
  let r = testQuatNorm(200000);
  print(r);
}

run();
