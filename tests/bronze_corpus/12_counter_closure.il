module D:/projects/brass/tests/bronze_corpus/12_counter_closure.js

func makeCounter(%0: f64) -> dynamic {
  b0:
    %1: dynamic = const.undefined
    %2: dynamic = env.create %1, 1
    env.init.tdz %2, 0, 0
    %3: dynamic = box.f64 %0
    env.set %2, 0, 0, %3
    %4: dynamic = create.func @__anon_fn_2, 1, %2
    ret %4
}

func run() -> void {
  b0:
    %0: f64 = const.f64 10
    %1: dynamic = call @makeCounter(%0)
    %2: f64 = const.f64 100
    %3: dynamic = call @makeCounter(%2)
    %4: dynamic = const.undefined
    %5: f64 = const.f64 5
    %6: dynamic = box.f64 %5
    %7: dynamic = call.dynamic %1, %4, 1, %6
    %8: dynamic = const.undefined
    %9: f64 = const.f64 3
    %10: dynamic = box.f64 %9
    %11: dynamic = call.dynamic %1, %8, 1, %10
    %12: dynamic = const.undefined
    %13: f64 = const.f64 20
    %14: dynamic = box.f64 %13
    %15: dynamic = call.dynamic %3, %12, 1, %14
    %16: dynamic = const.undefined
    %17: f64 = const.f64 30
    %18: dynamic = box.f64 %17
    %19: dynamic = call.dynamic %3, %16, 1, %18
    %20: dynamic = name.resolve "print"
    %21: dynamic = const.undefined
    %22: dynamic = call.dynamic %20, %21, 4, %7, %11, %15, %19
    ret
}

func __anon_fn_2(%0: dynamic, %1: dynamic) -> dynamic {
  b0:
    %2: dynamic = env.get.tdz %0, 0, 0, "count"
    %3: dynamic = add %2, %1
    %4: dynamic = env.get.tdz %0, 0, 0, "count"
    env.set %0, 0, 0, %3
    %5: dynamic = env.get.tdz %0, 0, 0, "count"
    ret %5
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
