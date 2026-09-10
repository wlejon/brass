module D:/projects/brass/tests/bronze_corpus/32_loop_fusion_contraction.js

func computeMapReduce(%0: f64) -> f64 {
  b0:
    %1: dynamic = create.array 100
    %2: f64 = const.f64 0
    jump b1(%2)
  b1(%3: f64):
    %4: bool = cmp.lt %3, %0
    br %4, b2, b3
  b2:
    %5: dynamic = box.f64 %3
    %6: dynamic = box.f64 %3
    elem.set %1, %5, %6, 0
    %7: f64 = const.f64 1
    %8: f64 = add %3, %7
    jump b1(%8)
  b3:
    %9: f64 = const.f64 0
    jump b4(%9)
  b4(%10: f64):
    %11: bool = cmp.lt %10, %0
    br %11, b5, b6
  b5:
    %12: dynamic = box.f64 %10
    %13: dynamic = elem.get %1, %12
    %14: f64 = unbox.f64 %13
    %15: f64 = const.f64 3
    %16: f64 = mul %14, %15
    %17: f64 = const.f64 7
    %18: f64 = add %16, %17
    %19: dynamic = box.f64 %18
    elem.set %1, %12, %19, 0
    %20: f64 = const.f64 1
    %21: f64 = add %10, %20
    jump b4(%21)
  b6:
    %22: f64 = const.f64 0
    %23: f64 = const.f64 0
    jump b7(%22, %23)
  b7(%24: f64, %25: f64):
    %26: bool = cmp.lt %25, %0
    br %26, b8, b9
  b8:
    %27: dynamic = box.f64 %25
    %28: dynamic = elem.get %1, %27
    %29: f64 = unbox.f64 %28
    %30: f64 = add %24, %29
    %31: f64 = const.f64 1
    %32: f64 = add %25, %31
    jump b7(%30, %32)
  b9:
    ret %24
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 1000
    %5: bool = cmp.lt %2, %4
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 100
    %7: f64 = call @computeMapReduce(%6)
    %8: f64 = add %3, %7
    %9: f64 = const.f64 1
    %10: f64 = add %2, %9
    jump b1(%10, %8)
  b3:
    %11: dynamic = name.resolve "print"
    %12: dynamic = const.undefined
    %13: dynamic = box.f64 %3
    %14: dynamic = call.dynamic %11, %12, 1, %13
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
