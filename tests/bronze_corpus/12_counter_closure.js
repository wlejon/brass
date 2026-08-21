function makeCounter(init) {
  let count = init;
  return function(step) {
    count = count + step;
    return count;
  };
}
function run() {
  let c1 = makeCounter(10);
  let c2 = makeCounter(100);
  let v1 = c1(5);
  let v2 = c1(3);
  let v3 = c2(20);
  let v4 = c2(30);
  print(v1, v2, v3, v4);
}
run();
