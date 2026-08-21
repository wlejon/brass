module D:/projects/brass/tests/bronze_corpus/02_bitwise.js

func bitTest(%0: f64, %1: f64) -> f64 {
  b0:
    %2: i32 = to.int32 %0
    %3: i32 = to.int32 %1
    %4: f64 = and %2, %3
    %5: i32 = to.int32 %0
    %6: i32 = to.int32 %1
    %7: f64 = or %5, %6
    %8: i32 = to.int32 %0
    %9: i32 = to.int32 %1
    %10: f64 = xor %8, %9
    %11: i32 = to.int32 %0
    %12: i32 = to.int32 %1
    %13: f64 = shl %11, %12
    %14: i32 = to.int32 %0
    %15: i32 = to.int32 %1
    %16: f64 = shr %14, %15
    %17: i32 = to.int32 %0
    %18: i32 = to.int32 %1
    %19: f64 = ushr %17, %18
    %20: f64 = add %4, %7
    %21: f64 = add %20, %10
    %22: f64 = add %21, %13
    %23: f64 = add %22, %16
    %24: f64 = add %23, %19
    ret %24
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: f64 = const.f64 6e+06
    %5: bool = cmp.lt %3, %4
    br %5, b2, b3(%2, %3)
  b2:
    %8: f64 = const.f64 12345
    %9: f64 = const.f64 5
    %10: f64 = call @bitTest(%8, %9)
    %11: f64 = const.f64 1
    %12: f64 = add %3, %11
    jump b1(%10, %12)
  b3(%6: f64, %7: f64):
    %13: dynamic = name.resolve "print"
    %14: dynamic = const.undefined
    %15: dynamic = box.f64 %6
    %16: dynamic = call.dynamic %13, %14, 1, %15
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
