function hotLoop(n) {
  let acc = 0;
  let i = 0;
  while (i < n) {
    acc = acc + (i * 3 + 1);
    i = i + 1;
  }
  return acc;
}

function run() {
  let res = hotLoop(1000);
  print(res);
}

run();
