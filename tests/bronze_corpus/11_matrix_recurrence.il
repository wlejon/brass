module D:/projects/brass/tests/bronze_corpus/11_matrix_recurrence.js

func recurrence2D(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3(%3, %4)
  b2:
    %8: f64 = const.f64 0
    jump b4(%3, %8)
  b3(%6: f64, %7: f64):
    ret %6
  b4(%9: f64, %10: f64):
    %11: bool = cmp.lt %10, %0
    br %11, b5, b6(%9, %10)
  b5:
    %14: f64 = const.f64 3
    %15: f64 = mul %4, %14
    %16: f64 = const.f64 7
    %17: f64 = mul %10, %16
    %18: f64 = add %15, %17
    %19: f64 = const.f64 1
    %20: f64 = add %18, %19
    %21: f64 = add %9, %20
    %22: f64 = const.f64 1000000007
    %23: f64 = mod %21, %22
    %24: f64 = const.f64 1
    %25: f64 = add %10, %24
    jump b4(%23, %25)
  b6(%12: f64, %13: f64):
    %26: f64 = const.f64 1
    %27: f64 = add %4, %26
    jump b1(%12, %27)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 5
    %1: f64 = call @recurrence2D(%0)
    %2: f64 = const.f64 10
    %3: f64 = call @recurrence2D(%2)
    %4: f64 = const.f64 6000
    %5: f64 = call @recurrence2D(%4)
    %6: dynamic = name.resolve "print"
    %7: dynamic = const.undefined
    %8: dynamic = box.f64 %1
    %9: dynamic = box.f64 %3
    %10: dynamic = box.f64 %5
    %11: dynamic = call.dynamic %6, %7, 3, %8, %9, %10
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
