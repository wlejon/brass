module D:/projects/brass/tests/bronze_corpus/29_generational_churn.js

func makeItem(%0: f64) -> dynamic {
  b0:
    %1: dynamic = create.object
    %2: dynamic = box.f64 %0
    prop.set %1, "val", %2
    ret %1
}

func run() -> void {
  b0:
    %0: dynamic = create.object
    %1: dynamic = const.undefined
    prop.set %0, "head", %1
    %2: f64 = const.f64 0
    %3: dynamic = box.f64 %2
    prop.set %0, "total", %3
    %4: f64 = const.f64 0
    jump b1(%4)
  b1(%5: f64):
    %6: f64 = const.f64 1000
    %7: bool = cmp.lt %5, %6
    br %7, b2, b6
  b2:
    %8: dynamic = call @makeItem(%5)
    %9: f64 = const.f64 2
    %10: f64 = mul %5, %9
    %11: dynamic = call @makeItem(%10)
    %12: dynamic = prop.get %8, "val"
    %13: dynamic = prop.get %11, "val"
    %14: dynamic = add %12, %13
    %15: f64 = const.f64 100
    %16: bool = cmp.lt %5, %15
    br %16, b3, b4
  b3:
    %17: dynamic = create.object
    prop.set %17, "val", %14
    %18: dynamic = prop.get %0, "head"
    prop.set %17, "next", %18
    prop.set %0, "head", %17
    %19: dynamic = prop.get %0, "total"
    %20: dynamic = add %19, %14
    prop.set %0, "total", %20
    jump b4
  b4:
    %21: f64 = const.f64 1
    %22: f64 = add %5, %21
    jump b1(%22)
  b6:
    %23: dynamic = prop.get %0, "total"
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
