function isPrime(n) {
  if (n < 2) return 0;
  let d = 2;
  while (d * d <= n) {
    if (n % d === 0) return 0;
    d = d + 1;
  }
  return 1;
}
function countPrimes(limit) {
  let count = 0;
  let i = 2;
  while (i <= limit) {
    if (isPrime(i) === 1) {
      count = count + 1;
    }
    i = i + 1;
  }
  return count;
}
function run() {
  let c100 = countPrimes(100);
  let c500 = countPrimes(500);
  print(c100, c500);
}
run();
