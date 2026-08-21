function gcd(a, b) {
  while (b > 0) {
    let t = b;
    b = a % b;
    a = t;
  }
  return a;
}
function run() {
  let g1 = gcd(1071, 462);
  let g2 = gcd(123456, 7890);
  print(g1, g2);
}
run();
