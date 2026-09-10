module D:/projects/brass/tests/bronze_corpus/37_fuzz_hardened_kernel.js

func hardened_step(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = const.f64 37
    %3: f64 = mul %0, %2
    %4: f64 = const.f64 13
    %5: f64 = mul %1, %4
    %6: f64 = add %3, %5
    %7: f64 = const.f64 7
    %8: f64 = add %6, %7
    %9: f64 = const.f64 65537
    %10: f64 = mod %8, %9
    %11: i32 = to.int32 %0
    %12: i32 = to.int32 %1
    %13: f64 = xor %11, %12
    %14: i32 = to.int32 %13
    %15: f64 = const.f64 65535
    %16: i32 = to.int32 %15
    %17: f64 = and %14, %16
    %18: f64 = add %10, %17
    %19: f64 = mod %18, %9
    ret %19
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
    %6: f64 = call @hardened_step(%4, %3)
    %7: f64 = add %3, %6
    %8: f64 = const.f64 65537
    %9: f64 = mod %7, %8
    %10: f64 = const.f64 1
    %11: f64 = add %4, %10
    jump b1(%9, %11)
  b3:
    ret %3
}

func run() -> void {
  b0:
    %0: f64 = const.f64 500
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
