function testOverflow() {
  let n = 9007199254740992;
  let i = 0;
  while (i < 100000000) {
    n = n + 1;
    i = i + 1;
  }
  return n;
}
function run() {
  print(testOverflow());
}
run();
