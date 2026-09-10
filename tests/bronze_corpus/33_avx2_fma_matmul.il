module D:/projects/brass/tests/bronze_corpus/33_avx2_fma_matmul.js

func matmulDot(%0: f64) -> f64 {
  b0:
    %1: dynamic = create.array 128
    %2: dynamic = create.array 128
    %3: f64 = const.f64 0
    jump b1(%3)
  b1(%4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 7
    %7: f64 = mod %4, %6
    %8: f64 = const.f64 1
    %9: f64 = add %7, %8
    %10: dynamic = box.f64 %4
    %11: dynamic = box.f64 %9
    elem.set %1, %10, %11, 0

    %12: f64 = const.f64 5
    %13: f64 = mod %4, %12
    %14: f64 = const.f64 2
    %15: f64 = add %13, %14
    %16: dynamic = box.f64 %15
    elem.set %2, %10, %16, 0

    %17: f64 = const.f64 1
    %18: f64 = add %4, %17
    jump b1(%18)
  b3:
    %19: f64 = const.f64 0
    %20: f64 = const.f64 0
    jump b4(%19, %20)
  b4(%21: f64, %22: f64):
    %23: bool = cmp.lt %22, %0
    br %23, b5, b6
  b5:
    %24: dynamic = box.f64 %22
    %25: dynamic = elem.get %1, %24
    %26: f64 = unbox.f64 %25
    %27: dynamic = elem.get %2, %24
    %28: f64 = unbox.f64 %27
    %29: f64 = mul %26, %28
    %30: f64 = add %21, %29
    %31: f64 = const.f64 1
    %32: f64 = add %22, %31
    jump b4(%30, %32)
  b6:
    ret %21
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 100
    %5: bool = cmp.lt %2, %4
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 128
    %7: f64 = call @matmulDot(%6)
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
