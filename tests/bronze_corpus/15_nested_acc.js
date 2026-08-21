function nestedAcc(n) {
  let outerSum = 0;
  let i = 0;
  while (i < n) {
    let j = 0;
    let innerAcc = 1;
    while (j < n) {
      innerAcc = (innerAcc * 3 + i + j) % 10007;
      outerSum = (outerSum + innerAcc) % 1000000007;
      j = j + 1;
    }
    outerSum = (outerSum + i) % 1000000007;
    i = i + 1;
  }
  return outerSum;
}
function run() {
  let r5 = nestedAcc(5);
  let r10 = nestedAcc(10);
  print(r5, r10);
}
run();
