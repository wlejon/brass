function recurrence2D(n) {
  let sum = 0;
  let i = 0;
  while (i < n) {
    let j = 0;
    while (j < n) {
      sum = (sum + (i * 3 + j * 7 + 1)) % 1000000007;
      j = j + 1;
    }
    i = i + 1;
  }
  return sum;
}
function run() {
  let r5 = recurrence2D(5);
  let r10 = recurrence2D(10);
  let rLarge = recurrence2D(6000);
  print(r5, r10, rLarge);
}
run();
