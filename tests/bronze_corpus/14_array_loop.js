function arraySum(n) {
  let arr = [1, 2, 3, 4, 5, 6, 7, 8];
  let sum = 0;
  let i = 0;
  while (i < 8) {
    sum = sum + arr[i];
    i = i + 1;
  }
  return sum;
}
function run() {
  print(arraySum(8));
}
run();
