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
  let i = 0;
  while (i < 1200000) {
    c1(5);
    c1(3);
    c2(20);
    c2(30);
    i = i + 1;
  }
  let v1 = c1(0);
  let v2 = c2(0);
  print(v1, v2);
}
run();
