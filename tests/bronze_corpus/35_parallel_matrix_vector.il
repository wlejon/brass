module D:/projects/brass/tests/bronze_corpus/35_parallel_matrix_vector.js

func dotRow(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = const.f64 0
    %3: f64 = const.f64 0
    jump b1(%2, %3)
  b1(%4: f64, %5: f64):
    %6: bool = cmp.lt %5, %1
    br %6, b2, b3
  b2:
    %7: f64 = const.f64 3
    %8: f64 = mul %0, %7
    %9: f64 = const.f64 7
    %10: f64 = mul %5, %9
    %11: f64 = add %8, %10
    %12: f64 = const.f64 1
    %13: f64 = add %11, %12
    %14: f64 = const.f64 10007
    %15: f64 = mod %13, %14
    %16: f64 = const.f64 5
    %17: f64 = mul %5, %16
    %18: f64 = add %17, %7
    %19: f64 = mod %18, %14
    %20: f64 = mul %15, %19
    %21: f64 = add %4, %20
    %22: f64 = mod %21, %14
    %23: f64 = add %5, %12
    jump b1(%22, %23)
  b3:
    ret %4
}

func matVec(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = const.f64 0
    %3: f64 = const.f64 0
    jump b1(%2, %3)
  b1(%4: f64, %5: f64):
    %6: bool = cmp.lt %5, %0
    br %6, b2, b3
  b2:
    %7: f64 = call @dotRow(%5, %1)
    %8: f64 = add %4, %7
    %9: f64 = const.f64 10007
    %10: f64 = mod %8, %9
    %11: f64 = const.f64 1
    %12: f64 = add %5, %11
    jump b1(%10, %12)
  b3:
    ret %4
}

func run() -> void {
  b0:
    %0: f64 = const.f64 400
    %1: f64 = const.f64 100
    %2: f64 = call @matVec(%0, %1)
    %3: dynamic = name.resolve "print"
    %4: dynamic = const.undefined
    %5: dynamic = box.f64 %2
    %6: dynamic = call.dynamic %3, %4, 1, %5
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
