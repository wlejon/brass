module D:/projects/brass/tests/bronze_corpus/24_shapes_polymorphic.js

func makeA(%0: f64) -> dynamic {
  b0:
    %1: dynamic = create.object
    %2: f64 = const.f64 1
    %3: dynamic = box.f64 %2
    prop.set %1, "a", %3
    %4: dynamic = box.f64 %0
    prop.set %1, "val", %4
    ret %1
}

func makeB(%0: f64) -> dynamic {
  b0:
    %1: dynamic = create.object
    %2: f64 = const.f64 2
    %3: dynamic = box.f64 %2
    prop.set %1, "b", %3
    %4: f64 = const.f64 3
    %5: dynamic = box.f64 %4
    prop.set %1, "c", %5
    %6: dynamic = box.f64 %0
    prop.set %1, "val", %6
    ret %1
}

func makeC(%0: f64) -> dynamic {
  b0:
    %1: dynamic = create.object
    %2: dynamic = box.f64 %0
    prop.set %1, "val", %2
    ret %1
}

func getVal(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = prop.get %0, "val"
    ret %1
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 0
    %2: dynamic = box.f64 %1
    jump b1(%0, %2)
  b1(%3: f64, %4: dynamic):
    %5: f64 = const.f64 1000
    %6: bool = cmp.lt %3, %5
    br %6, b2, b3(%3, %4)
  b2:
    %7: dynamic = call @makeA(%3)
    %8: f64 = const.f64 2
    %9: f64 = mul %3, %8
    %10: dynamic = call @makeB(%9)
    %11: f64 = const.f64 3
    %12: f64 = mul %3, %11
    %13: dynamic = call @makeC(%12)
    %14: dynamic = call @getVal(%7)
    %15: dynamic = call @getVal(%10)
    %16: dynamic = call @getVal(%13)
    %17: dynamic = add %14, %15
    %18: dynamic = add %17, %16
    %19: dynamic = add %4, %18
    %20: f64 = const.f64 1
    %21: f64 = add %3, %20
    jump b1(%21, %19)
  b3(%22: f64, %23: dynamic):
    %24: dynamic = name.resolve "print"
    %25: dynamic = const.undefined
    %26: dynamic = call.dynamic %24, %25, 1, %23
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
