function bitTest(a, b) {
  let b_and = a & b;
  let b_or = a | b;
  let b_xor = a ^ b;
  let b_shl = a << b;
  let b_shr = a >> b;
  let b_ushr = a >>> b;
  return b_and + b_or + b_xor + b_shl + b_shr + b_ushr;
}
function run() {
  let r = 0;
  let i = 0;
  while (i < 6000000) {
    r = bitTest(12345, 5);
    i = i + 1;
  }
  print(r);
}
run();
