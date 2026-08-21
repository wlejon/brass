function collatz(n) {
  let steps = 0;
  while (n > 1) {
    if ((n % 2) === 0) {
      n = n / 2;
    } else {
      n = 3 * n + 1;
    }
    steps = steps + 1;
  }
  return steps;
}
function collatzRange(limit) {
  let total = 0;
  let i = 1;
  while (i <= limit) {
    total = (total + collatz(i)) % 1000000007;
    i = i + 1;
  }
  return total;
}
function run() {
  let s27 = collatz(27);
  let s12 = collatz(12);
  let s1 = collatz(1);
  let total = collatzRange(150000);
  print(s27, s12, s1, total);
}
run();
