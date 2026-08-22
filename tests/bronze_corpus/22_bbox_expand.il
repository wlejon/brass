module D:/projects/brass/tests/bronze_corpus/22_bbox_expand.js

func Vector3(%0: f64, %1: f64, %2: f64) -> dynamic {
  b0:
    %3: dynamic = create.object
    %4: dynamic = box.f64 %0
    prop.set %3, 6, %4, 0, 0
    %5: dynamic = box.f64 %1
    prop.set %3, 7, %5, 1, 0
    %6: dynamic = box.f64 %2
    prop.set %3, 8, %6, 2, 0
    ret %3
}

func Box3(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = create.object
    prop.set %2, 9, %0, 3, 0
    prop.set %2, 10, %1, 4, 0
    ret %2
}

func expandByPoint(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = prop.get %0, 9, 5, mono
    %3: dynamic = prop.get %0, 10, 6, mono
    %4: dynamic = prop.get %1, 6, 7
    %5: dynamic = prop.get %2, 6, 8
    %6: bool = rel.lt %4, %5
    br %6, b1, b2
  b1:
    %7: dynamic = prop.get %1, 6, 9
    prop.set %2, 6, %7, 10, 0
    jump b3
  b2:
    jump b3
  b3:
    %8: dynamic = prop.get %1, 6, 11
    %9: dynamic = prop.get %3, 6, 12
    %10: bool = rel.gt %8, %9
    br %10, b4, b5
  b4:
    %11: dynamic = prop.get %1, 6, 13
    prop.set %3, 6, %11, 14, 0
    jump b6
  b5:
    jump b6
  b6:
    %12: dynamic = prop.get %1, 7, 15
    %13: dynamic = prop.get %2, 7, 16
    %14: bool = rel.lt %12, %13
    br %14, b7, b8
  b7:
    %15: dynamic = prop.get %1, 7, 17
    prop.set %2, 7, %15, 18, 0
    jump b9
  b8:
    jump b9
  b9:
    %16: dynamic = prop.get %1, 7, 19
    %17: dynamic = prop.get %3, 7, 20
    %18: bool = rel.gt %16, %17
    br %18, b10, b11
  b10:
    %19: dynamic = prop.get %1, 7, 21
    prop.set %3, 7, %19, 22, 0
    jump b12
  b11:
    jump b12
  b12:
    %20: dynamic = prop.get %1, 8, 23
    %21: dynamic = prop.get %2, 8, 24
    %22: bool = rel.lt %20, %21
    br %22, b13, b14
  b13:
    %23: dynamic = prop.get %1, 8, 25
    prop.set %2, 8, %23, 26, 0
    jump b15
  b14:
    jump b15
  b15:
    %24: dynamic = prop.get %1, 8, 27
    %25: dynamic = prop.get %3, 8, 28
    %26: bool = rel.gt %24, %25
    br %26, b16, b17
  b16:
    %27: dynamic = prop.get %1, 8, 29
    prop.set %3, 8, %27, 30, 0
    jump b18
  b17:
    jump b18
  b18:
    ret %0
}

func unionBox(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = prop.get %1, 9, 31, mono
    %3: dynamic = call @expandByPoint(%0, %2)
    %4: dynamic = prop.get %1, 10, 32, mono
    %5: dynamic = call @expandByPoint(%0, %4)
    ret %0
}

func testBBoxExpand(%0: f64, %1: f64) -> dynamic {
  b0:
    %2: dynamic = create.array 0
    %3: f64 = const.f64 0
    jump b1(%3)
  b1(%4: f64):
    %5: bool = cmp.lt %4, %1
    br %5, b2, b3(%4)
  b2:
    %7: dynamic = box.f64 %4
    %8: f64 = const.f64 0.15
    %9: f64 = mul %4, %8
    %10: f64 = const.f64 25
    %11: f64 = sub %9, %10
    %12: f64 = const.f64 0.25
    %13: f64 = mul %4, %12
    %14: f64 = const.f64 35
    %15: f64 = sub %13, %14
    %16: f64 = const.f64 0.35
    %17: f64 = mul %4, %16
    %18: f64 = const.f64 45
    %19: f64 = sub %17, %18
    %20: dynamic = call @Vector3(%11, %15, %19)
    elem.set %2, %7, %20, 0
    %21: f64 = const.f64 1
    %22: f64 = add %4, %21
    jump b1(%22)
  b3(%6: f64):
    %23: f64 = const.f64 0
    %24: f64 = const.f64 0
    %25: f64 = const.f64 0
    %26: dynamic = call @Vector3(%23, %24, %25)
    %27: f64 = const.f64 0
    %28: f64 = const.f64 0
    %29: f64 = const.f64 0
    %30: dynamic = call @Vector3(%27, %28, %29)
    %31: dynamic = call @Box3(%26, %30)
    %32: f64 = const.f64 0
    jump b4(%32)
  b4(%33: f64):
    %34: bool = cmp.lt %33, %0
    br %34, b5, b6(%33)
  b5:
    %36: f64 = const.f64 1000
    %37: f64 = const.f64 1000
    %38: f64 = const.f64 1000
    %39: dynamic = call @Vector3(%36, %37, %38)
    %40: f64 = const.f64 1000
    %41: f64 = neg %40
    %42: f64 = const.f64 1000
    %43: f64 = neg %42
    %44: f64 = const.f64 1000
    %45: f64 = neg %44
    %46: dynamic = call @Vector3(%41, %43, %45)
    %47: dynamic = call @Box3(%39, %46)
    %48: f64 = const.f64 0
    jump b7(%48)
  b6(%35: f64):
    %67: dynamic = prop.get %31, 10, 35, mono
    %68: dynamic = prop.get %67, 6, 36
    %69: dynamic = prop.get %31, 9, 37, mono
    %70: dynamic = prop.get %69, 6, 38
    %71: dynamic = sub %68, %70
    %72: dynamic = prop.get %31, 10, 39, mono
    %73: dynamic = prop.get %72, 7, 40
    %74: dynamic = prop.get %31, 9, 41, mono
    %75: dynamic = prop.get %74, 7, 42
    %76: dynamic = sub %73, %75
    %77: dynamic = prop.get %31, 10, 43, mono
    %78: dynamic = prop.get %77, 8, 44
    %79: dynamic = prop.get %31, 9, 45, mono
    %80: dynamic = prop.get %79, 8, 46
    %81: dynamic = sub %78, %80
    %82: dynamic = add %71, %76
    %83: dynamic = add %82, %81
    ret %83
  b7(%49: f64):
    %50: bool = cmp.lt %49, %1
    br %50, b8, b9(%49)
  b8:
    %52: dynamic = box.f64 %49
    %53: dynamic = elem.get %2, %52
    %54: dynamic = call @expandByPoint(%47, %53)
    %55: f64 = const.f64 1
    %56: f64 = add %49, %55
    jump b7(%56)
  b9(%51: f64):
    %57: dynamic = call @unionBox(%31, %47)
    %58: f64 = mod %33, %1
    %59: dynamic = box.f64 %58
    %60: dynamic = elem.get %2, %59
    %61: dynamic = prop.get %60, 6, 33
    %62: f64 = const.f64 1e-05
    %63: dynamic = box.f64 %62
    %64: dynamic = add %61, %63
    prop.set %60, 6, %64, 34, 0
    %65: f64 = const.f64 1
    %66: f64 = add %33, %65
    jump b4(%66)
}

func run() -> void {
  b0:
    %0: f64 = const.f64 30000
    %1: f64 = const.f64 50
    %2: dynamic = call @testBBoxExpand(%0, %1)
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
