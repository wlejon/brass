function fibRec(n) {
  if (n < 2) return n;
  return fibRec(n - 1) + fibRec(n - 2);
}
function run() {
  let f5 = fibRec(5);
  let f10 = fibRec(10);
  let f15 = fibRec(15);
  let f34 = fibRec(34);
  print(f5, f10, f15, f34);
}
run();
