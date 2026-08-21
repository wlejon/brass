function sqrtApprox(x) {
  let g = x / 2;
  let i = 0;
  while (i < 20) {
    g = (g + x / g) / 2;
    i = i + 1;
  }
  return g;
}
function run() {
  let s = 0;
  let i = 0;
  while (i < 1000000) {
    s = s + sqrtApprox(625);
    i = i + 1;
  }
  let s144 = sqrtApprox(144);
  let s256 = sqrtApprox(256);
  print(s144, s256, s);
}
run();
