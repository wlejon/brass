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
    %4: dynamic = const.undefined
    %5: f64 = const.f64 10
    %6: dynamic = box.f64 %5
    %7: dynamic = call.dynamic %1, %4, 1, %6
    %8: dynamic = const.undefined
    %9: f64 = const.f64 10
    %10: dynamic = box.f64 %9
    %11: dynamic = call.dynamic %3, %8, 1, %10
    %12: dynamic = name.resolve "print"
    %13: dynamic = const.undefined
    %14: dynamic = call.dynamic %12, %13, 2, %7, %11
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
