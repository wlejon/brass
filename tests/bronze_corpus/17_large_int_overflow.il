module D:/projects/brass/tests/bronze_corpus/17_large_int_overflow.js

func testOverflow() -> f64 {
  b0:
    %0: f64 = const.f64 9007199254740992
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 10
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = const.f64 1
    %9: f64 = add %2, %8
    %10: f64 = const.f64 1
    %11: f64 = add %3, %10
    jump b1(%9, %11)
  b3(%6: f64, %7: f64):
    ret %6
}

func run() -> void {
  b0:
    %0: dynamic = name.resolve "print"
    %1: dynamic = const.undefined
    %2: f64 = call @testOverflow()
    %3: dynamic = box.f64 %2
    %4: dynamic = call.dynamic %0, %1, 1, %3
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
