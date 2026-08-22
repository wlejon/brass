module D:/projects/brass/tests/bronze_corpus/19_vec3_acc.js

func Vector3(%0: f64, %1: f64, %2: f64) -> dynamic {
  b0:
    %3: dynamic = create.object
    %4: dynamic = box.f64 %0
    prop.set %3, 5, %4, 0, 0
    %5: dynamic = box.f64 %1
    prop.set %3, 6, %5, 1, 0
    %6: dynamic = box.f64 %2
    prop.set %3, 7, %6, 2, 0
    ret %3
}

func addVectors(%0: dynamic, %1: dynamic, %2: dynamic) -> dynamic {
  b0:
    %3: dynamic = prop.get %1, 5, 3, mono
    %4: dynamic = prop.get %2, 5, 4
    %5: dynamic = add %3, %4
    prop.set %0, 5, %5, 5, 0
    %6: dynamic = prop.get %1, 6, 6, mono
    %7: dynamic = prop.get %2, 6, 7
    %8: dynamic = add %6, %7
    prop.set %0, 6, %8, 8, 0
    %9: dynamic = prop.get %1, 7, 9, mono
    %10: dynamic = prop.get %2, 7, 10
    %11: dynamic = add %9, %10
    prop.set %0, 7, %11, 11, 0
    ret %0
}

func multiplyScalar(%0: dynamic, %1: f64) -> dynamic {
  b0:
    %2: dynamic = prop.get %0, 5, 12, mono
    %3: dynamic = box.f64 %1
    %4: dynamic = mul %2, %3
    prop.set %0, 5, %4, 13, 0
    %5: dynamic = prop.get %0, 6, 14, mono
    %6: dynamic = box.f64 %1
    %7: dynamic = mul %5, %6
    prop.set %0, 6, %7, 15, 0
    %8: dynamic = prop.get %0, 7, 16, mono
    %9: dynamic = box.f64 %1
    %10: dynamic = mul %8, %9
    prop.set %0, 7, %10, 17, 0
    ret %0
}

func testVec3Acc(%0: f64, %1: f64) -> dynamic {
  b0:
    %2: dynamic = create.array 0
    %3: f64 = const.f64 0
    jump b1(%3)
  b1(%4: f64):
    %5: bool = cmp.lt %4, %1
    br %5, b2, b3(%4)
  b2:
    %7: dynamic = box.f64 %4
    %8: f64 = const.f64 0.1
    %9: f64 = mul %4, %8
    %10: f64 = const.f64 0.2
    %11: f64 = mul %4, %10
    %12: f64 = const.f64 0.3
    %13: f64 = mul %4, %12
    %14: dynamic = call @Vector3(%9, %11, %13)
    elem.set %2, %7, %14, 0
    %15: f64 = const.f64 1
    %16: f64 = add %4, %15
    jump b1(%16)
  b3(%6: f64):
    %17: f64 = const.f64 0
    %18: f64 = const.f64 0
    %19: f64 = const.f64 0
    %20: dynamic = call @Vector3(%17, %18, %19)
    %21: f64 = const.f64 0
    jump b4(%21)
  b4(%22: f64):
    %23: bool = cmp.lt %22, %0
    br %23, b5, b6(%22)
  b5:
    %25: f64 = const.f64 0
    jump b7(%25)
  b6(%24: f64):
    %38: dynamic = prop.get %20, 5, 18, mono
    %39: dynamic = prop.get %20, 6, 19, mono
    %40: dynamic = add %38, %39
    %41: dynamic = prop.get %20, 7, 20, mono
    %42: dynamic = add %40, %41
    ret %42
  b7(%26: f64):
    %27: bool = cmp.lt %26, %1
    br %27, b8, b9(%26)
  b8:
    %29: dynamic = box.f64 %26
    %30: dynamic = elem.get %2, %29
    %31: dynamic = call @addVectors(%20, %20, %30)
    %32: f64 = const.f64 0.999999
    %33: dynamic = call @multiplyScalar(%20, %32)
    %34: f64 = const.f64 1
    %35: f64 = add %26, %34
    jump b7(%35)
  b9(%28: f64):
    %36: f64 = const.f64 1
    %37: f64 = add %22, %36
    jump b4(%37)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 20000
    %1: f64 = const.f64 100
    %2: dynamic = call @testVec3Acc(%0, %1)
    %3: dynamic = name.resolve "print"
    %4: dynamic = const.undefined
    %5: dynamic = call.dynamic %3, %4, 1, %2
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
