function branchDiamond(cond, x, y) {
  let val = 0;
  if (cond > 0) {
    val = x * 7 + y;
  } else {
    val = y * 2;
  }
  let extra = x * 7 + y;
  return val + extra;
}

function loopDiamonds(n) {
  let acc = 0;
  let i = 0;
  let c = 5;
  let d = 9;
  while (i < n) {
    let inv = c * 3 + d;
    let b = i % 2;
    let branchRes = 0;
    if (b > 0) {
      branchRes = i * 4 + inv;
    } else {
      branchRes = i + 1;
    }
    let post = i * 4 + inv;
    acc = acc + branchRes + post;
    i = i + 1;
  }
  return acc;
}

function run() {
  let total = 0;
  let k = 0;
  while (k < 100) {
    let d1 = branchDiamond(k % 2, k, 13);
    total = total + d1;
    k = k + 1;
  }
  let loopTotal = loopDiamonds(100);
  let finalResult = total + loopTotal;
  print(finalResult);
}

run();
