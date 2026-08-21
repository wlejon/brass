module D:/projects/brass/tests/bronze_corpus/07_prime_count.js

func isPrime(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 2
    %2: bool = cmp.lt %0, %1
    br %2, b1, b2
  b1:
    %3: f64 = const.f64 0
    ret %3
  b2:
    jump b3
  b3:
    %4: f64 = const.f64 2
    jump b4(%4)
  b4(%5: f64):
    %6: f64 = mul %5, %5
    %7: bool = cmp.le %6, %0
    br %7, b5, b6(%5)
  b5:
    %9: f64 = mod %0, %5
    %10: f64 = const.f64 0
    %11: bool = cmp.eq %9, %10
    br %11, b7, b8
  b6(%8: f64):
    %15: f64 = const.f64 1
    ret %15
  b7:
    %12: f64 = const.f64 0
    ret %12
  b8:
    jump b9
  b9:
    %13: f64 = const.f64 1
    %14: f64 = add %5, %13
    jump b4(%14)
}

func countPrimes(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 2
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.le %4, %0
    br %5, b2, b3(%3, %4)
  b2:
    %8: f64 = call @isPrime(%4)
    %9: f64 = const.f64 1
    %10: bool = cmp.eq %8, %9
    br %10, b4, b5
  b3(%6: f64, %7: f64):
    ret %6
  b4:
    %11: f64 = const.f64 1
    %12: f64 = add %3, %11
    jump b6(%12)
  b5:
    jump b6(%3)
  b6(%13: f64):
    %14: f64 = const.f64 1
    %15: f64 = add %4, %14
    jump b1(%13, %15)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 100
    %1: f64 = call @countPrimes(%0)
    %2: f64 = const.f64 500
    %3: f64 = call @countPrimes(%2)
    %4: dynamic = name.resolve "print"
    %5: dynamic = const.undefined
    %6: dynamic = box.f64 %1
    %7: dynamic = box.f64 %3
    %8: dynamic = call.dynamic %4, %5, 2, %6, %7
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
