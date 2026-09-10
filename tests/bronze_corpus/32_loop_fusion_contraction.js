function computeMapReduce(n) {
  let arr = [];
  let i = 0;
  while (i < n) {
    arr[i] = i;
    i = i + 1;
  }
  let j = 0;
  while (j < n) {
    arr[j] = arr[j] * 3 + 7;
    j = j + 1;
  }
  let sum = 0;
  let m = 0;
  while (m < n) {
    sum = sum + arr[m];
    m = m + 1;
  }
  return sum;
}

function run() {
  let k = 0;
  let total = 0;
  while (k < 1000) {
    total = total + computeMapReduce(100);
    k = k + 1;
  }
  print(total);
}

run();
