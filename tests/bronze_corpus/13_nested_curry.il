module D:/projects/brass/tests/bronze_corpus/13_nested_curry.js

func curried3(%0: f64) -> dynamic {
  b0:
    %1: dynamic = const.undefined
    %2: dynamic = env.create %1, 1
    %3: dynamic = box.f64 %0
    env.set %2, 0, 0, %3
    %4: dynamic = create.func @__anon_fn_2, 1, %2
    ret %4
}

func run() -> void {
  b0:
    %0: f64 = const.f64 5
    %1: dynamic = call @curried3(%0)
    %2: dynamic = const.undefined
    %3: f64 = const.f64 4
    %4: dynamic = box.f64 %3
    %5: dynamic = call.dynamic %1, %2, 1, %4
    %6: f64 = const.f64 0
    %7: f64 = const.f64 0
    %10: dynamic = box.f64 %6
    jump b1(%10, %7)
  b1(%8: dynamic, %9: f64):
    %11: f64 = const.f64 2e+05
    %12: bool = cmp.lt %9, %11
    br %12, b2, b3(%8, %9)
  b2:
    %15: dynamic = const.undefined
    %16: f64 = const.f64 3
    %17: dynamic = box.f64 %16
    %18: dynamic = call.dynamic %5, %15, 1, %17
    %19: dynamic = add %8, %18
    %20: f64 = const.f64 9
    %21: dynamic = call @curried3(%20)
    %22: dynamic = const.undefined
    %23: f64 = const.f64 8
    %24: dynamic = box.f64 %23
    %25: dynamic = call.dynamic %21, %22, 1, %24
    %26: dynamic = const.undefined
    %27: f64 = const.f64 7
    %28: dynamic = box.f64 %27
    %29: dynamic = call.dynamic %25, %26, 1, %28
    %30: dynamic = add %19, %29
    %31: f64 = const.f64 1
    %32: f64 = add %9, %31
    jump b1(%30, %32)
  b3(%13: dynamic, %14: f64):
    %33: dynamic = const.undefined
    %34: f64 = const.f64 3
    %35: dynamic = box.f64 %34
    %36: dynamic = call.dynamic %5, %33, 1, %35
    %37: f64 = const.f64 9
    %38: dynamic = call @curried3(%37)
    %39: dynamic = const.undefined
    %40: f64 = const.f64 8
    %41: dynamic = box.f64 %40
    %42: dynamic = call.dynamic %38, %39, 1, %41
    %43: dynamic = const.undefined
    %44: f64 = const.f64 7
    %45: dynamic = box.f64 %44
    %46: dynamic = call.dynamic %42, %43, 1, %45
    %47: dynamic = name.resolve "print"
    %48: dynamic = const.undefined
    %49: dynamic = call.dynamic %47, %48, 3, %36, %46, %13
    ret
}

func __anon_fn_2(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.get %0, 1, 0
    %3: f64 = unbox.f64 %2
    %4: f64 = const.f64 100
    %5: f64 = mul %3, %4
    %6: dynamic = env.get %0, 0, 0
    %7: f64 = const.f64 10
    %8: dynamic = box.f64 %7
    %9: dynamic = mul %6, %8
    %10: dynamic = box.f64 %5
    %11: dynamic = add %10, %9
    %12: dynamic = add %11, %1
    ret %12
}

func __anon_fn_2(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.create %0, 1
    env.set %2, 0, 0, %1
    %3: dynamic = create.func @__anon_fn_2, 1, %2
    ret %3
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
