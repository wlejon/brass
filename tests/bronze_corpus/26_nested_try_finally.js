function step(x) {
  if (x === 1) throw 10;
  if (x === 2) throw 20;
  return 1;
}

function test(x) {
  let res = 0;
  try {
    let inner_res = 0;
    try {
      inner_res = step(x);
    } catch (e) {
      if (e === 10) {
        inner_res = 100;
      } else {
        throw e;
      }
    }
    res = inner_res + 5;
  } catch (e2) {
    res = 5 + e2 * 2;
  }
  return res;
}

function run() {
  let total = 0;
  total += test(0);
  total += test(1);
  total += test(2);
  print(total);
}

run();
