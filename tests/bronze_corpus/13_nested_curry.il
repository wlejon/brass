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
    %6: dynamic = const.undefined
    %7: f64 = const.f64 3
    %8: dynamic = box.f64 %7
    %9: dynamic = call.dynamic %5, %6, 1, %8
    %10: f64 = const.f64 9
    %11: dynamic = call @curried3(%10)
    %12: dynamic = const.undefined
    %13: f64 = const.f64 8
    %14: dynamic = box.f64 %13
    %15: dynamic = call.dynamic %11, %12, 1, %14
    %16: dynamic = const.undefined
    %17: f64 = const.f64 7
    %18: dynamic = box.f64 %17
    %19: dynamic = call.dynamic %15, %16, 1, %18
    %20: dynamic = name.resolve "print"
    %21: dynamic = const.undefined
    %22: dynamic = call.dynamic %20, %21, 2, %9, %19
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
