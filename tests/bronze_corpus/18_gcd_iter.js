function gcd(a, b) {
  while (b > 0) {
    let t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function run() {
  let total = 0;
  let i = 0;
  while (i < 1200000) {
    total = total + gcd(1071 + (i % 100), 462 + (i % 50)) + gcd(48 + (i % 20), 18 + (i % 10));
    i = i + 1;
  }
  print(total);
}

run();
