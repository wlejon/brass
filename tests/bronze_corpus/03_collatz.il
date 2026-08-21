module D:/projects/brass/tests/bronze_corpus/03_collatz.js

func collatz(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 1
    %5: bool = cmp.gt %2, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = const.f64 2
    %9: f64 = mod %2, %8
    %10: f64 = const.f64 0
    %11: bool = cmp.eq %9, %10
    br %11, b4, b5
  b3(%6: f64, %7: f64):
    ret %7
  b4:
    %12: f64 = const.f64 2
    %13: f64 = div %2, %12
    jump b6(%13)
  b5:
    %14: f64 = const.f64 3
    %15: f64 = mul %14, %2
    %16: f64 = const.f64 1
    %17: f64 = add %15, %16
    jump b6(%17)
  b6(%18: f64):
    %19: f64 = const.f64 1
    %20: f64 = add %3, %19
    jump b1(%18, %20)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 27
    %1: f64 = call @collatz(%0)
    %2: f64 = const.f64 12
    %3: f64 = call @collatz(%2)
    %4: f64 = const.f64 1
    %5: f64 = call @collatz(%4)
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
