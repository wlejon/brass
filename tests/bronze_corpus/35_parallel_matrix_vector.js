function dotRow(i, cols) {
  let sum = 0;
  let j = 0;
  while (j < cols) {
    let a_ij = (i * 3 + j * 7 + 1) % 10007;
    let v_j = (j * 5 + 3) % 10007;
    sum = (sum + a_ij * v_j) % 10007;
    j = j + 1;
  }
  return sum;
}

function matVec(rows, cols) {
  let total = 0;
  let i = 0;
  while (i < rows) {
    let row_val = dotRow(i, cols);
    total = (total + row_val) % 10007;
    i = i + 1;
  }
  return total;
}

function run() {
  let res = matVec(400, 100);
  print(res);
}

run();
