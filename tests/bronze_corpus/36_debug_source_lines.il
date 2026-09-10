module D:/projects/brass/tests/bronze_corpus/36_debug_source_lines.js

func stepA(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 17
    %2: f64 = mul %0, %1
    %3: f64 = const.f64 5
    %4: f64 = add %2, %3
    %5: f64 = const.f64 10007
    %6: f64 = mod %4, %5
    ret %6
}

func stepB(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = call @stepA(%0)
    %3: f64 = call @stepA(%1)
    %4: f64 = add %2, %3
    %5: f64 = const.f64 10007
    %6: f64 = mod %4, %5
    ret %6
}

func compute(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 0
    jump b1(%1, %2)
  b1(%3: f64, %4: f64):
    %5: bool = cmp.lt %4, %0
    br %5, b2, b3
  b2:
    %6: f64 = const.f64 2
    %7: f64 = mul %4, %6
    %8: f64 = call @stepB(%4, %7)
    %9: f64 = add %3, %8
    %10: f64 = const.f64 10007
    %11: f64 = mod %9, %10
    %12: f64 = const.f64 1
    %13: f64 = add %4, %12
    jump b1(%11, %13)
  b3:
    ret %3
}

func run() -> void {
  b0:
    %0: f64 = const.f64 100
    %1: f64 = call @compute(%0)
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