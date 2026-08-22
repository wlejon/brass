module D:/projects/brass/tests/bronze_corpus/18_gcd_iter.js

func gcd(%0: f64, %1: f64) -> f64 {
  b0:
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 0
    %5: bool = cmp.gt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = mod %2, %3
    jump b1(%3, %8)
  b3(%6: f64, %7: f64):
    ret %6
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 1200000
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = const.f64 1071
    %9: f64 = const.f64 100
    %10: f64 = mod %3, %9
    %11: f64 = add %8, %10
    %12: f64 = const.f64 462
    %13: f64 = const.f64 50
    %14: f64 = mod %3, %13
    %15: f64 = add %12, %14
    %16: f64 = call @gcd(%11, %15)
    %17: f64 = add %2, %16
    %18: f64 = const.f64 48
    %19: f64 = const.f64 20
    %20: f64 = mod %3, %19
    %21: f64 = add %18, %20
    %22: f64 = const.f64 18
    %23: f64 = const.f64 10
    %24: f64 = mod %3, %23
    %25: f64 = add %22, %24
    %26: f64 = call @gcd(%21, %25)
    %27: f64 = add %17, %26
    %28: f64 = const.f64 1
    %29: f64 = add %3, %28
    jump b1(%27, %29)
  b3(%6: f64, %7: f64):
    %30: dynamic = name.resolve "print"
    %31: dynamic = const.undefined
    %32: dynamic = box.f64 %6
    %33: dynamic = call.dynamic %30, %31, 1, %32
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
