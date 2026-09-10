async function step1(x) {
  return x + 10;
}

async function step2(x) {
  return x * 2;
}

async function step3(x) {
  return x + 5;
}

async function run() {
  let val = 5;
  let a = await step1(val);
  let b = await step2(a);
  let c = await step3(b);
  print(c);
}

run();
