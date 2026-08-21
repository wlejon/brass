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
  let s144 = sqrtApprox(144);
  let s256 = sqrtApprox(256);
  let s625 = sqrtApprox(625);
  print(s144, s256, s625);
}
run();
