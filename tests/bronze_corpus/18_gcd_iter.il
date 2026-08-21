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
    %0: f64 = const.f64 1071
    %1: f64 = const.f64 462
    %2: f64 = call @gcd(%0, %1)
    %3: f64 = const.f64 123456
    %4: f64 = const.f64 7890
    %5: f64 = call @gcd(%3, %4)
    %6: dynamic = name.resolve "print"
    %7: dynamic = const.undefined
    %8: dynamic = box.f64 %2
    %9: dynamic = box.f64 %5
    %10: dynamic = call.dynamic %6, %7, 2, %8, %9
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
