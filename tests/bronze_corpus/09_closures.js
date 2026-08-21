function makeMultiplier(factor) {
  return function(val) {
    return val * factor;
  };
}
function run() {
  let mul3 = makeMultiplier(3);
  let mul7 = makeMultiplier(7);
  let sum = 0;
  let i = 0;
  while (i < 4500000) {
    sum = sum + mul3(10) + mul7(10);
    i = i + 1;
  }
  let r1 = mul3(10);
  let r2 = mul7(10);
  print(r1, r2, sum);
}
run();
