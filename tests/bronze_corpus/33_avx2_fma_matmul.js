function matmulDot(n) {
  let a = [];
  let b = [];
  let i = 0;
  while (i < n) {
    a[i] = (i % 7) + 1;
    b[i] = (i % 5) + 2;
    i = i + 1;
  }
  let sum = 0;
  let k = 0;
  while (k < n) {
    sum = sum + a[k] * b[k];
    k = k + 1;
  }
  return sum;
}

function run() {
  let iter = 0;
  let total = 0;
  while (iter < 100) {
    total = total + matmulDot(128);
    iter = iter + 1;
  }
  print(total);
}

run();
