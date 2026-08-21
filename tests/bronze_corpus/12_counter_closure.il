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
    %4: f64 = const.f64 0
    jump b1(%4)
  b1(%5: f64):
    %6: f64 = const.f64 1200000
    %7: bool = cmp.lt %5, %6
    br %7, b2, b3(%5)
  b2:
    %9: dynamic = const.undefined
    %10: f64 = const.f64 5
    %11: dynamic = box.f64 %10
    %12: dynamic = call.dynamic %1, %9, 1, %11
    %13: dynamic = const.undefined
    %14: f64 = const.f64 3
    %15: dynamic = box.f64 %14
    %16: dynamic = call.dynamic %1, %13, 1, %15
    %17: dynamic = const.undefined
    %18: f64 = const.f64 20
    %19: dynamic = box.f64 %18
    %20: dynamic = call.dynamic %3, %17, 1, %19
    %21: dynamic = const.undefined
    %22: f64 = const.f64 30
    %23: dynamic = box.f64 %22
    %24: dynamic = call.dynamic %3, %21, 1, %23
    %25: f64 = const.f64 1
    %26: f64 = add %5, %25
    jump b1(%26)
  b3(%8: f64):
    %27: dynamic = const.undefined
    %28: f64 = const.f64 0
    %29: dynamic = box.f64 %28
    %30: dynamic = call.dynamic %1, %27, 1, %29
    %31: dynamic = const.undefined
    %32: f64 = const.f64 0
    %33: dynamic = box.f64 %32
    %34: dynamic = call.dynamic %3, %31, 1, %33
    %35: dynamic = name.resolve "print"
    %36: dynamic = const.undefined
    %37: dynamic = call.dynamic %35, %36, 2, %30, %34
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
