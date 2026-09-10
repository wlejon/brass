function risky(x) {
  if (x > 5) {
    throw x * 2;
  }
  return x + 1;
}

function run() {
  let sum = 0;
  for (let i = 0; i < 10; i++) {
    try {
      sum += risky(i);
    } catch (e) {
      sum += e;
    }
  }
  print(sum);
}

run();
