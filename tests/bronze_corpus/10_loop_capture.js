function loopCapture() {
  let sum = 0;
  for (let i = 1; i <= 300000; i = i + 1) {
    let fn = function() { return i * 10; };
    sum = sum + fn();
  }
  print(sum);
}
loopCapture();
