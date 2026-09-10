module D:/projects/brass/tests/bronze_corpus/30_osr_hot_loop.js

func hotLoop(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 3
    %7: f64 = mul %4, %6
    %8: f64 = const.f64 1
    %9: f64 = add %7, %8
    %10: f64 = add %3, %9
    %11: f64 = const.f64 1
    %12: f64 = add %4, %11
    jump b1(%10, %12)
  b3:
    ret %3
}

func run() -> void {
  b0:
    %0: f64 = const.f64 1000
    %1: f64 = call @hotLoop(%0)
    %2: dynamic = name.resolve "print"
    %3: dynamic = const.undefined
    %4: dynamic = box.f64 %1
    %5: dynamic = call.dynamic %2, %3, 1, %4
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
