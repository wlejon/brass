module D:/projects/brass/tests/bronze_corpus/31_gvn_pre_diamonds.js

func branchDiamond(%0: f64, %1: f64, %2: f64) -> f64 {
  b0:
    %3: f64 = const.f64 0
    %4: bool = cmp.gt %0, %3
    br %4, b1, b2
  b1:
    %5: f64 = const.f64 7
    %6: f64 = mul %1, %5
    %7: f64 = add %6, %2
    jump b3(%7)
  b2:
    %8: f64 = const.f64 2
    %9: f64 = mul %2, %8
    jump b3(%9)
  b3(%10: f64):
    %11: f64 = const.f64 7
    %12: f64 = mul %1, %11
    %13: f64 = add %12, %2
    %14: f64 = add %10, %13
    ret %14
}

func loopDiamonds(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 5
    %7: f64 = const.f64 3
    %8: f64 = mul %6, %7
    %9: f64 = const.f64 9
    %10: f64 = add %8, %9
    %11: f64 = const.f64 2
    %12: f64 = mod %4, %11
    %13: f64 = const.f64 0
    %14: bool = cmp.gt %12, %13
    br %14, b4, b5
  b4:
    %15: f64 = const.f64 4
    %16: f64 = mul %4, %15
    %17: f64 = add %16, %10
    jump b6(%17)
  b5:
    %18: f64 = const.f64 1
    %19: f64 = add %4, %18
    jump b6(%19)
  b6(%20: f64):
    %21: f64 = const.f64 4
    %22: f64 = mul %4, %21
    %23: f64 = add %22, %10
    %24: f64 = add %3, %20
    %25: f64 = add %24, %23
    %26: f64 = const.f64 1
    %27: f64 = add %4, %26
    jump b1(%25, %27)
  b3:
    ret %3
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 100
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 2
    %7: f64 = mod %3, %6
    %8: f64 = const.f64 13
    %9: f64 = call @branchDiamond(%7, %3, %8)
    %10: f64 = add %2, %9
    %11: f64 = const.f64 1
    %12: f64 = add %3, %11
    jump b1(%10, %12)
  b3:
    %13: f64 = const.f64 100
    %14: f64 = call @loopDiamonds(%13)
    %15: f64 = add %2, %14
    %16: dynamic = name.resolve "print"
    %17: dynamic = const.undefined
    %18: dynamic = box.f64 %15
    %19: dynamic = call.dynamic %16, %17, 1, %18
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
