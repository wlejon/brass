function computeStep(a, b) {
  let x = (a * 3 + b * 2 + 7) % 10007;
  let y = (x * 5 + a + 11) % 10007;
  return (x + y) % 10007;
}

function run() {
  let total = 0;
  let i = 0;
  while (i < 300) {
    total = (total + computeStep(i, total)) % 10007;
    i = i + 1;
  }
  print(total);
}

run();
