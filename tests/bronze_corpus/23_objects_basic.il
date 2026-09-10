module D:/projects/brass/tests/bronze_corpus/23_objects_basic.js

func makePoint(%0: f64, %1: f64) -> dynamic {
  b0:
    %2: dynamic = create.object
    %3: dynamic = box.f64 %0
    prop.set %2, "x", %3
    %4: dynamic = box.f64 %1
    prop.set %2, "y", %4
    ret %2
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
    %7: f64 = const.f64 2
    %8: f64 = mul %3, %7
    %9: dynamic = call @makePoint(%3, %8)
    %10: dynamic = prop.get %9, "x"
    %11: dynamic = prop.get %9, "y"
    %12: dynamic = add %10, %11
    prop.set %9, "z", %12
    %13: dynamic = prop.get %9, "z"
    %14: dynamic = add %4, %13
    %15: f64 = const.f64 1
    %16: f64 = add %3, %15
    jump b1(%16, %14)
  b3(%17: f64, %18: dynamic):
    %19: dynamic = name.resolve "print"
    %20: dynamic = const.undefined
    %21: dynamic = call.dynamic %19, %20, 1, %18
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
