function mathTest(a, b) {
  let add = a + b;
  let sub = a - b;
  let mul = a * b;
  let div = a / b;
  let rem = a % b;
  let neg = -a;
  return add + sub + mul + div + rem + neg;
}
function run() {
  let r1 = mathTest(100, 25);
  let r2 = mathTest(7, 3);
  print(r1, r2);
}
run();
