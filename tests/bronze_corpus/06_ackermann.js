function ack(m, n) {
  if (m === 0) return n + 1;
  if (n === 0) return ack(m - 1, 1);
  return ack(m - 1, ack(m, n - 1));
}
function run() {
  let a1 = ack(2, 4);
  let a2 = ack(3, 2);
  let a3 = ack(3, 3);
  print(a1, a2, a3);
}
run();
