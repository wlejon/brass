module D:/projects/brass/tests/bronze_corpus/16_param_bounds.js

func rangeSum(%0: f64, %1: f64, %2: f64, %3: f64) -> f64 {
  b0:
    %4: f64 = const.f64 0
    jump b1(%4, %0)
  b1(%5: f64, %6: f64):
    %7: bool = cmp.lt %6, %1
    br %7, b2, b3(%5, %6)
  b2:
    %10: f64 = mul %6, %3
    %11: f64 = add %5, %10
    %12: f64 = add %6, %2
    jump b1(%11, %12)
  b3(%8: f64, %9: f64):
    ret %8
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 100
    %2: f64 = const.f64 2
    %3: f64 = const.f64 3
    %4: f64 = call @rangeSum(%0, %1, %2, %3)
    %5: f64 = const.f64 10
    %6: f64 = const.f64 50
    %7: f64 = const.f64 5
    %8: f64 = const.f64 7
    %9: f64 = call @rangeSum(%5, %6, %7, %8)
    %10: dynamic = name.resolve "print"
    %11: dynamic = const.undefined
    %12: dynamic = box.f64 %4
    %13: dynamic = box.f64 %9
    %14: dynamic = call.dynamic %10, %11, 2, %12, %13
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
