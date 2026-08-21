module D:/projects/brass/tests/bronze_corpus/09_closures.js

func makeMultiplier(%0: f64) -> dynamic {
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
    %0: f64 = const.f64 3
    %1: dynamic = call @makeMultiplier(%0)
    %2: f64 = const.f64 7
    %3: dynamic = call @makeMultiplier(%2)
    %4: f64 = const.f64 0
    %5: f64 = const.f64 0
    %8: dynamic = box.f64 %4
    jump b1(%8, %5)
  b1(%6: dynamic, %7: f64):
    %9: f64 = const.f64 4500000
    %10: bool = cmp.lt %7, %9
    br %10, b2, b3(%6, %7)
  b2:
    %13: dynamic = const.undefined
    %14: f64 = const.f64 10
    %15: dynamic = box.f64 %14
    %16: dynamic = call.dynamic %1, %13, 1, %15
    %17: dynamic = add %6, %16
    %18: dynamic = const.undefined
    %19: f64 = const.f64 10
    %20: dynamic = box.f64 %19
    %21: dynamic = call.dynamic %3, %18, 1, %20
    %22: dynamic = add %17, %21
    %23: f64 = const.f64 1
    %24: f64 = add %7, %23
    jump b1(%22, %24)
  b3(%11: dynamic, %12: f64):
    %25: dynamic = const.undefined
    %26: f64 = const.f64 10
    %27: dynamic = box.f64 %26
    %28: dynamic = call.dynamic %1, %25, 1, %27
    %29: dynamic = const.undefined
    %30: f64 = const.f64 10
    %31: dynamic = box.f64 %30
    %32: dynamic = call.dynamic %3, %29, 1, %31
    %33: dynamic = name.resolve "print"
    %34: dynamic = const.undefined
    %35: dynamic = call.dynamic %33, %34, 3, %28, %32, %11
    ret
}

func __anon_fn_2(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.get %0, 0, 0
    %3: f64 = unbox.f64 %2
    %4: dynamic = box.f64 %3
    %5: dynamic = mul %1, %4
    ret %5
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
