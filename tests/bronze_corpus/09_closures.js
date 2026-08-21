function makeMultiplier(factor) {
  return function(val) {
    return val * factor;
  };
}
function run() {
  let mul3 = makeMultiplier(3);
  let mul7 = makeMultiplier(7);
  let r1 = mul3(10);
  let r2 = mul7(10);
  print(r1, r2);
}
run();
