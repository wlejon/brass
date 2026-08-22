module D:/projects/brass/tests/bronze_corpus/21_quat_norm.js

func Quaternion(%0: f64, %1: f64, %2: f64, %3: f64) -> dynamic {
  b0:
    %4: dynamic = create.object
    %5: dynamic = box.f64 %0
    prop.set %4, 6, %5, 0, 0
    %6: dynamic = box.f64 %1
    prop.set %4, 7, %6, 1, 0
    %7: dynamic = box.f64 %2
    prop.set %4, 8, %7, 2, 0
    %8: dynamic = box.f64 %3
    prop.set %4, 9, %8, 3, 0
    ret %4
}

func sqrt(%0: dynamic) -> dynamic {
  b0:
    %1: f64 = const.f64 0
    %2: dynamic = box.f64 %1
    %3: bool = rel.le %0, %2
    br %3, b1, b2
  b1:
    %4: f64 = const.f64 0
    %5: dynamic = box.f64 %4
    ret %5
  b2:
    jump b3
  b3:
    %6: f64 = const.f64 0
    jump b4(%0, %6)
  b4(%7: dynamic, %8: f64):
    %9: f64 = const.f64 12
    %10: bool = cmp.lt %8, %9
    br %10, b5, b6(%7, %8)
  b5:
    %13: f64 = const.f64 0.5
    %14: dynamic = div %0, %7
    %15: dynamic = add %7, %14
    %16: dynamic = box.f64 %13
    %17: dynamic = mul %16, %15
    %18: f64 = const.f64 1
    %19: f64 = add %8, %18
    jump b4(%17, %19)
  b6(%11: dynamic, %12: f64):
    ret %11
}

func quatNormalize(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = prop.get %0, 6, 4, mono
    %2: dynamic = prop.get %0, 6, 5, mono
    %3: dynamic = mul %1, %2
    %4: dynamic = prop.get %0, 7, 6, mono
    %5: dynamic = prop.get %0, 7, 7, mono
    %6: dynamic = mul %4, %5
    %7: dynamic = add %3, %6
    %8: dynamic = prop.get %0, 8, 8, mono
    %9: dynamic = prop.get %0, 8, 9, mono
    %10: dynamic = mul %8, %9
    %11: dynamic = add %7, %10
    %12: dynamic = prop.get %0, 9, 10, mono
    %13: dynamic = prop.get %0, 9, 11, mono
    %14: dynamic = mul %12, %13
    %15: dynamic = add %11, %14
    %16: f64 = const.f64 0
    %18: dynamic = box.f64 %16
    %17: bool = strict.eq %15, %18
    br %17, b1, b2
  b1:
    %19: f64 = const.f64 0
    %20: dynamic = box.f64 %19
    prop.set %0, 6, %20, 12, 0
    %21: f64 = const.f64 0
    %22: dynamic = box.f64 %21
    prop.set %0, 7, %22, 13, 0
    %23: f64 = const.f64 0
    %24: dynamic = box.f64 %23
    prop.set %0, 8, %24, 14, 0
    %25: f64 = const.f64 1
    %26: dynamic = box.f64 %25
    prop.set %0, 9, %26, 15, 0
    ret %0
  b2:
    jump b3
  b3:
    %27: f64 = const.f64 1
    %28: dynamic = call @sqrt(%15)
    %29: dynamic = box.f64 %27
    %30: dynamic = div %29, %28
    %31: dynamic = prop.get %0, 6, 16, mono
    %32: dynamic = mul %31, %30
    prop.set %0, 6, %32, 17, 0
    %33: dynamic = prop.get %0, 7, 18, mono
    %34: dynamic = mul %33, %30
    prop.set %0, 7, %34, 19, 0
    %35: dynamic = prop.get %0, 8, 20, mono
    %36: dynamic = mul %35, %30
    prop.set %0, 8, %36, 21, 0
    %37: dynamic = prop.get %0, 9, 22, mono
    %38: dynamic = mul %37, %30
    prop.set %0, 9, %38, 23, 0
    ret %0
}

func quatMultiply(%0: dynamic, %1: dynamic, %2: dynamic) -> dynamic {
  b0:
    %3: dynamic = prop.get %1, 6, 24, mono
    %4: dynamic = prop.get %1, 7, 25, mono
    %5: dynamic = prop.get %1, 8, 26, mono
    %6: dynamic = prop.get %1, 9, 27, mono
    %7: dynamic = prop.get %2, 6, 28, mono
    %8: dynamic = prop.get %2, 7, 29, mono
    %9: dynamic = prop.get %2, 8, 30, mono
    %10: dynamic = prop.get %2, 9, 31, mono
    %11: dynamic = mul %3, %10
    %12: dynamic = mul %6, %7
    %13: dynamic = add %11, %12
    %14: dynamic = mul %4, %9
    %15: dynamic = add %13, %14
    %16: dynamic = mul %5, %8
    %17: dynamic = sub %15, %16
    prop.set %0, 6, %17, 32, 0
    %18: dynamic = mul %4, %10
    %19: dynamic = mul %6, %8
    %20: dynamic = add %18, %19
    %21: dynamic = mul %5, %7
    %22: dynamic = add %20, %21
    %23: dynamic = mul %3, %9
    %24: dynamic = sub %22, %23
    prop.set %0, 7, %24, 33, 0
    %25: dynamic = mul %5, %10
    %26: dynamic = mul %6, %9
    %27: dynamic = add %25, %26
    %28: dynamic = mul %3, %8
    %29: dynamic = add %27, %28
    %30: dynamic = mul %4, %7
    %31: dynamic = sub %29, %30
    prop.set %0, 8, %31, 34, 0
    %32: dynamic = mul %6, %10
    %33: dynamic = mul %3, %7
    %34: dynamic = sub %32, %33
    %35: dynamic = mul %4, %8
    %36: dynamic = sub %34, %35
    %37: dynamic = mul %5, %9
    %38: dynamic = sub %36, %37
    prop.set %0, 9, %38, 35, 0
    ret %0
}

func testQuatNorm(%0: f64) -> dynamic {
  b0:
    %1: f64 = const.f64 0.1
    %2: f64 = const.f64 0.2
    %3: f64 = const.f64 0.3
    %4: f64 = const.f64 0.9
    %5: dynamic = call @Quaternion(%1, %2, %3, %4)
    %6: f64 = const.f64 0.01
    %7: f64 = const.f64 0.02
    %8: f64 = const.f64 0.03
    %9: f64 = const.f64 0.99
    %10: dynamic = call @Quaternion(%6, %7, %8, %9)
    %11: f64 = const.f64 0
    %12: f64 = const.f64 0
    %13: f64 = const.f64 0
    %14: f64 = const.f64 1
    %15: dynamic = call @Quaternion(%11, %12, %13, %14)
    %16: f64 = const.f64 0
    jump b1(%16)
  b1(%17: f64):
    %18: bool = cmp.lt %17, %0
    br %18, b2, b3(%17)
  b2:
    %20: dynamic = call @quatMultiply(%15, %15, %5)
    %21: dynamic = call @quatNormalize(%15)
    %22: dynamic = call @quatMultiply(%15, %15, %10)
    %23: dynamic = call @quatNormalize(%15)
    %24: dynamic = prop.get %5, 6, 36, mono
    %25: f64 = const.f64 1e-07
    %26: dynamic = box.f64 %25
    %27: dynamic = add %24, %26
    prop.set %5, 6, %27, 37, 0
    %28: f64 = const.f64 1
    %29: f64 = add %17, %28
    jump b1(%29)
  b3(%19: f64):
    %30: dynamic = prop.get %15, 6, 38, mono
    %31: dynamic = prop.get %15, 7, 39, mono
    %32: dynamic = add %30, %31
    %33: dynamic = prop.get %15, 8, 40, mono
    %34: dynamic = add %32, %33
    %35: dynamic = prop.get %15, 9, 41, mono
    %36: dynamic = add %34, %35
    ret %36
}

func run() -> void {
  b0:
    %0: f64 = const.f64 2e+05
    %1: dynamic = call @testQuatNorm(%0)
    %2: dynamic = name.resolve "print"
    %3: dynamic = const.undefined
    %4: dynamic = call.dynamic %2, %3, 1, %1
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
