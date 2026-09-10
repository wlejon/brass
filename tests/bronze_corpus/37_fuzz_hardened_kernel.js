function hardened_step(x, y) {
  let a = (x * 37 + y * 13 + 7) % 65537;
  let b = (x ^ y) & 65535;
  let c = (a + b) % 65537;
  return c;
}

function compute(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    let s = hardened_step(i, acc);
    acc = (acc + s) % 65537;
    i = i + 1;
  }
  return acc;
}

function run() {
  let ans = compute(500);
  print(ans);
}

run();
