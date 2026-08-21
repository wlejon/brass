function fibIter(n) {
  let a = 0;
  let b = 1;
  let i = 0;
  while (i < n) {
    let t = (a + b) % 1000000007;
    a = b;
    b = t;
    i = i + 1;
  }
  return a;
}
function run() {
  let f0 = fibIter(0);
  let f1 = fibIter(1);
  let f10 = fibIter(10);
  let f20 = fibIter(20);
  let f30 = fibIter(30);
  let fLarge = fibIter(25000000);
  print(f0, f1, f10, f20, f30, fLarge);
}
run();
