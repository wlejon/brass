function stepA(x) {
  let res = (x * 17 + 5) % 10007;
  return res;
}

function stepB(y, z) {
  let val = stepA(y) + stepA(z);
  return val % 10007;
}

function compute(n) {
  let total = 0;
  let i = 0;
  while (i < n) {
    let s = stepB(i, i * 2);
    total = (total + s) % 10007;
    i = i + 1;
  }
  return total;
}

function run() {
  let ans = compute(100);
  print(ans);
}

run();