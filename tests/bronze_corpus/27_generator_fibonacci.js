function* fibonacci() {
  let a = 0;
  let b = 1;
  while (true) {
    yield a;
    let next = a + b;
    a = b;
    b = next;
  }
}

function run() {
  let gen = fibonacci();
  let iter = gen;
  let sum = 0;
  for (let i = 0; i < 10; i++) {
    let step = iter.next();
    sum += step.value;
  }
  print(sum);
}

run();
