module D:/projects/brass/tests/bronze_corpus/34_background_tiering.js

func computeStep(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = const.f64 3
    %3: f64 = mul %0, %2
    %4: f64 = const.f64 2
    %5: f64 = mul %1, %4
    %6: f64 = add %3, %5
    %7: f64 = const.f64 7
    %8: f64 = add %6, %7
    %9: f64 = const.f64 10007
    %10: f64 = mod %8, %9
    %11: f64 = const.f64 5
    %12: f64 = mul %10, %11
    %13: f64 = add %12, %0
    %14: f64 = const.f64 11
    %15: f64 = add %13, %14
    %16: f64 = mod %15, %9
    %17: f64 = add %10, %16
    %18: f64 = mod %17, %9
    ret %18
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 300
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = call @computeStep(%3, %2)
    %9: f64 = add %2, %8
    %10: f64 = const.f64 10007
    %11: f64 = mod %9, %10
    %12: f64 = const.f64 1
    %13: f64 = add %3, %12
    jump b1(%11, %13)
  b3(%6: f64, %7: f64):
    %14: dynamic = name.resolve "print"
    %15: dynamic = const.undefined
    %16: dynamic = box.f64 %6
    %17: dynamic = call.dynamic %14, %15, 1, %16
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
