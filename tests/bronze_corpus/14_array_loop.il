module D:/projects/brass/tests/bronze_corpus/14_array_loop.js

func arraySum(%0: f64) -> dynamic {
  b0:
    %1: dynamic = create.array 8
    %2: f64 = const.f64 1
    %3: dynamic = box.f64 %2
    prop.set %1, 2, %3, 0, 0
    %4: f64 = const.f64 2
    %5: dynamic = box.f64 %4
    prop.set %1, 3, %5, 1, 0
    %6: f64 = const.f64 3
    %7: dynamic = box.f64 %6
    prop.set %1, 4, %7, 2, 0
    %8: f64 = const.f64 4
    %9: dynamic = box.f64 %8
    prop.set %1, 5, %9, 3, 0
    %10: f64 = const.f64 5
    %11: dynamic = box.f64 %10
    prop.set %1, 6, %11, 4, 0
    %12: f64 = const.f64 6
    %13: dynamic = box.f64 %12
    prop.set %1, 7, %13, 5, 0
    %14: f64 = const.f64 7
    %15: dynamic = box.f64 %14
    prop.set %1, 8, %15, 6, 0
    %16: f64 = const.f64 8
    %17: dynamic = box.f64 %16
    prop.set %1, 9, %17, 7, 0
    %18: f64 = const.f64 0
    %19: f64 = const.f64 0
    %22: dynamic = box.f64 %18
    jump b1(%22, %19)
  b1(%20: dynamic, %21: f64):
    %23: f64 = const.f64 8
    %24: bool = cmp.lt %21, %23
    br %24, b2, b3(%20, %21)
  b2:
    %27: dynamic = box.f64 %21
    %28: dynamic = elem.get %1, %27
    %29: dynamic = add %20, %28
    %30: f64 = const.f64 1
    %31: f64 = add %21, %30
    jump b1(%29, %31)
  b3(%25: dynamic, %26: f64):
    ret %25
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    %4: dynamic = box.f64 %1
    jump b1(%0, %4)
  b1(%2: f64, %3: dynamic):
    %5: f64 = const.f64 1e+06
    %6: bool = cmp.lt %2, %5
    br %6, b2, b3(%2, %3)
  b2:
    %9: f64 = const.f64 8
    %10: dynamic = call @arraySum(%9)
    %11: dynamic = add %3, %10
    %12: f64 = const.f64 1
    %13: f64 = add %2, %12
    jump b1(%13, %11)
  b3(%7: f64, %8: dynamic):
    %14: dynamic = name.resolve "print"
    %15: dynamic = const.undefined
    %16: dynamic = call.dynamic %14, %15, 1, %8
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
