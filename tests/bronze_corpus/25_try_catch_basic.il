module tests/bronze_corpus/25_try_catch_basic.js

func risky(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 5
    %2: bool = cmp.gt %0, %1
    br %2, b1, b2
  b1:
    %3: f64 = const.f64 2
    %4: f64 = mul %0, %3
    %5: dynamic = box.f64 %4
    throw %5
  b2:
    %6: f64 = const.f64 1
    %7: f64 = add %0, %6
    ret %7
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 10
    %5: bool = cmp.lt %3, %4
    br %5, b2, b5(%2)
  b2 handler b4:
    %6: f64 = call @risky(%3)
    %7: f64 = add %2, %6
    jump b3(%7, %3)
  b4:
    %8: dynamic = exc.take
    %9: f64 = unbox.f64 %8
    %10: f64 = add %2, %9
    jump b3(%10, %3)
  b3(%11: f64, %12: f64):
    %13: f64 = const.f64 1
    %14: f64 = add %12, %13
    jump b1(%11, %14)
  b5(%15: f64):
    print %15
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
