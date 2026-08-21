module D:/projects/brass/tests/bronze_corpus/01_arithmetic.js

func mathTest(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = add %0, %1
    %3: f64 = sub %0, %1
    %4: f64 = mul %0, %1
    %5: f64 = div %0, %1
    %6: f64 = mod %0, %1
    %7: f64 = neg %0
    %8: f64 = add %2, %3
    %9: f64 = add %8, %4
    %10: f64 = add %9, %5
    %11: f64 = add %10, %6
    %12: f64 = add %11, %7
    ret %12
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%0, %1, %2)
  b1(%3: f64, %4: f64, %5: f64):
    %6: f64 = const.f64 4e+06
    %7: bool = cmp.lt %5, %6
    br %7, b2, b3(%3, %4, %5)
  b2:
    %11: f64 = const.f64 100
    %12: f64 = const.f64 25
    %13: f64 = call @mathTest(%11, %12)
    %14: f64 = const.f64 7
    %15: f64 = const.f64 3
    %16: f64 = call @mathTest(%14, %15)
    %17: f64 = const.f64 1
    %18: f64 = add %5, %17
    jump b1(%13, %16, %18)
  b3(%8: f64, %9: f64, %10: f64):
    %19: dynamic = name.resolve "print"
    %20: dynamic = const.undefined
    %21: dynamic = box.f64 %8
    %22: dynamic = box.f64 %9
    %23: dynamic = call.dynamic %19, %20, 2, %21, %22
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
