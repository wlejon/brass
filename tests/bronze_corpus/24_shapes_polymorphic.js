function makeA(v) {
  let obj = { a: 1, val: v };
  return obj;
}

function makeB(v) {
  let obj = { b: 2, c: 3, val: v };
  return obj;
}

function makeC(v) {
  let obj = { val: v };
  return obj;
}

function getVal(obj) {
  return obj.val;
}

function run() {
  let total = 0;
  let i = 0;
  while (i < 1000) {
    let a = makeA(i);
    let b = makeB(i * 2);
    let c = makeC(i * 3);
    total = total + getVal(a) + getVal(b) + getVal(c);
    i = i + 1;
  }
  print(total);
}

run();
