module D:/projects/brass/tests/bronze_corpus/06_ackermann.js

func ack(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = const.f64 0
    %3: bool = cmp.eq %0, %2
    br %3, b1, b2
  b1:
    %4: f64 = const.f64 1
    %5: f64 = add %1, %4
    ret %5
  b2:
    jump b3
  b3:
    %6: f64 = const.f64 0
    %7: bool = cmp.eq %1, %6
    br %7, b4, b5
  b4:
    %8: f64 = const.f64 1
    %9: f64 = sub %0, %8
    %10: f64 = const.f64 1
    %11: f64 = call @ack(%9, %10)
    ret %11
  b5:
    jump b6
  b6:
    %12: f64 = const.f64 1
    %13: f64 = sub %0, %12
    %14: f64 = const.f64 1
    %15: f64 = sub %1, %14
    %16: f64 = call @ack(%0, %15)
    %17: f64 = call @ack(%13, %16)
    ret %17
}

func run() -> void {
  b0:
    %0: f64 = const.f64 2
    %1: f64 = const.f64 4
    %2: f64 = call @ack(%0, %1)
    %3: f64 = const.f64 3
    %4: f64 = const.f64 2
    %5: f64 = call @ack(%3, %4)
    %6: f64 = const.f64 3
    %7: f64 = const.f64 3
    %8: f64 = call @ack(%6, %7)
    %9: dynamic = name.resolve "print"
    %10: dynamic = const.undefined
    %11: dynamic = box.f64 %2
    %12: dynamic = box.f64 %5
    %13: dynamic = box.f64 %8
    %14: dynamic = call.dynamic %9, %10, 3, %11, %12, %13
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
