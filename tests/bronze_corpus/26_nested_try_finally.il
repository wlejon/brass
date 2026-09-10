module tests/bronze_corpus/26_nested_try_finally.js

func step(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 1
    %2: bool = cmp.eq %0, %1
    br %2, b1, b2
  b1:
    %3: f64 = const.f64 10
    %4: dynamic = box.f64 %3
    throw %4
  b2:
    %5: f64 = const.f64 2
    %6: bool = cmp.eq %0, %5
    br %6, b3, b4
  b3:
    %7: f64 = const.f64 20
    %8: dynamic = box.f64 %7
    throw %8
  b4:
    %9: f64 = const.f64 1
    ret %9
}

func test(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    jump b1
  b1 handler b7:
    jump b2
  b2 handler b4:
    %2: f64 = call @step(%0)
    jump b3(%2)
  b4:
    %3: dynamic = exc.take
    %4: f64 = unbox.f64 %3
    %5: f64 = const.f64 10
    %6: bool = cmp.eq %4, %5
    br %6, b5, b6(%3)
  b5:
    %7: f64 = const.f64 100
    jump b3(%7)
  b6(%8: dynamic) handler b7:
    throw %8
  b3(%9: f64):
    %10: f64 = const.f64 5
    %11: f64 = add %9, %10
    ret %11
  b7:
    %12: dynamic = exc.take
    %13: f64 = unbox.f64 %12
    %14: f64 = const.f64 2
    %15: f64 = mul %13, %14
    %16: f64 = const.f64 5
    %17: f64 = add %16, %15
    ret %17
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = call @test(%0)
    %2: f64 = const.f64 1
    %3: f64 = call @test(%2)
    %4: f64 = const.f64 2
    %5: f64 = call @test(%4)
    %6: f64 = add %1, %3
    %7: f64 = add %6, %5
    print %7
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
