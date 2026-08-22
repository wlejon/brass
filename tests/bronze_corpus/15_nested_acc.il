module D:/projects/brass/tests/bronze_corpus/15_nested_acc.js

func nestedAcc(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3(%3, %4)
  b2:
    %8: f64 = const.f64 0
    %9: f64 = const.f64 1
    jump b4(%3, %8, %9)
  b3(%6: f64, %7: f64):
    ret %6
  b4(%10: f64, %11: f64, %12: f64):
    %13: bool = cmp.lt %11, %0
    br %13, b5, b6(%10, %11, %12)
  b5:
    %17: f64 = const.f64 3
    %18: f64 = mul %12, %17
    %19: f64 = add %18, %4
    %20: f64 = add %19, %11
    %21: f64 = const.f64 10007
    %22: f64 = mod %20, %21
    %23: f64 = add %10, %22
    %24: f64 = const.f64 1000000007
    %25: f64 = mod %23, %24
    %26: f64 = const.f64 1
    %27: f64 = add %11, %26
    jump b4(%25, %27, %22)
  b6(%14: f64, %15: f64, %16: f64):
    %28: f64 = add %14, %4
    %29: f64 = const.f64 1000000007
    %30: f64 = mod %28, %29
    %31: f64 = const.f64 1
    %32: f64 = add %4, %31
    jump b1(%30, %32)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 3000
    %1: f64 = call @nestedAcc(%0)
    %2: f64 = const.f64 4000
    %3: f64 = call @nestedAcc(%2)
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
