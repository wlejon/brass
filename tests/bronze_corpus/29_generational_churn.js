function makeItem(v) {
  let item = { val: v };
  return item;
}

function run() {
  let holder = { head: null, total: 0 };
  let i = 0;
  while (i < 1000) {
    let t1 = makeItem(i);
    let t2 = makeItem(i * 2);
    let sum = t1.val + t2.val;
    if (i < 100) {
      let node = { val: sum, next: holder.head };
      holder.head = node;
      holder.total = holder.total + sum;
    }
    i = i + 1;
  }
  print(holder.total);
}

run();
