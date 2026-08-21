module D:/projects/brass/tests/bronze_corpus/08_newton_sqrt.js

func sqrtApprox(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 2
    %2: f64 = div %0, %1
    %3: f64 = const.f64 0
    jump b1(%2, %3)
  b1(%4: f64, %5: f64):
    %6: f64 = const.f64 20
    %7: bool = cmp.lt %5, %6
    br %7, b2, b3(%4, %5)
  b2:
    %10: f64 = div %0, %4
    %11: f64 = add %4, %10
    %12: f64 = const.f64 2
    %13: f64 = div %11, %12
    %14: f64 = const.f64 1
    %15: f64 = add %5, %14
    jump b1(%13, %15)
  b3(%8: f64, %9: f64):
    ret %8
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 1e+06
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = const.f64 625
    %9: f64 = call @sqrtApprox(%8)
    %10: f64 = add %2, %9
    %11: f64 = const.f64 1
    %12: f64 = add %3, %11
    jump b1(%10, %12)
  b3(%6: f64, %7: f64):
    %13: f64 = const.f64 144
    %14: f64 = call @sqrtApprox(%13)
    %15: f64 = const.f64 256
    %16: f64 = call @sqrtApprox(%15)
    %17: dynamic = name.resolve "print"
    %18: dynamic = const.undefined
    %19: dynamic = box.f64 %14
    %20: dynamic = box.f64 %16
    %21: dynamic = box.f64 %6
    %22: dynamic = call.dynamic %17, %18, 3, %19, %20, %21
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
