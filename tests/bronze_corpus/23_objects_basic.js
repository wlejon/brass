function makePoint(x, y) {
  let pt = { x: x, y: y };
  return pt;
}

function run() {
  let i = 0;
  let total = 0;
  while (i < 1000) {
    let p = makePoint(i, i * 2);
    p.z = p.x + p.y;
    total = total + p.z;
    i = i + 1;
  }
  print(total);
}

run();
