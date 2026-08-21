function curried3(a) {
  return function(b) {
    return function(c) {
      return a * 100 + b * 10 + c;
    };
  };
}
function run() {
  let f1 = curried3(5);
  let f2 = f1(4);
  let r1 = f2(3);
  let r2 = curried3(9)(8)(7);
  print(r1, r2);
}
run();
