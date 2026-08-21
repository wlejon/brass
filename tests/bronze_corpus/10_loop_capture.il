module D:/projects/brass/tests/bronze_corpus/10_loop_capture.js

func loopCapture() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: dynamic = const.undefined
    %2: dynamic = env.create %1, 1
    env.init.tdz %2, 0, 0
    %3: f64 = const.f64 1
    %4: dynamic = box.f64 %3
    env.set %2, 0, 0, %4
    %5: dynamic = const.undefined
    %6: dynamic = env.create %5, 1
    %7: dynamic = env.get %2, 0, 0
    env.set %6, 0, 0, %7
    %10: dynamic = box.f64 %0
    jump b1(%10, %6)
  b1(%8: dynamic, %9: dynamic):
    %12: dynamic = env.get.tdz %9, 0, 0, "i"
    %13: f64 = unbox.f64 %12
    %14: f64 = const.f64 4
    %15: bool = cmp.le %13, %14
    br %15, b2, b4(%8)
  b2:
    %18: dynamic = create.func @__anon_fn_1, 0, %9
    %19: dynamic = const.undefined
    %20: dynamic = call.dynamic %18, %19, 0
    %21: dynamic = add %8, %20
    jump b3(%21, %9)
  b3(%16: dynamic, %17: dynamic):
    %22: dynamic = env.create %5, 1
    %23: dynamic = env.get %17, 0, 0
    env.set %22, 0, 0, %23
    %24: dynamic = env.get.tdz %22, 0, 0, "i"
    %25: f64 = unbox.f64 %24
    %26: f64 = const.f64 1
    %27: f64 = add %25, %26
    %28: dynamic = env.get.tdz %22, 0, 0, "i"
    %29: dynamic = box.f64 %27
    env.set %22, 0, 0, %29
    jump b1(%16, %22)
  b4(%11: dynamic):
    %30: dynamic = name.resolve "print"
    %31: dynamic = const.undefined
    %32: dynamic = call.dynamic %30, %31, 1, %11
    ret
}

func __anon_fn_1(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = env.get.tdz %0, 0, 0, "i"
    %2: f64 = unbox.f64 %1
    %3: f64 = const.f64 10
    %4: f64 = mul %2, %3
    %5: dynamic = box.f64 %4
    ret %5
}

func main() -> void {
  b0:
    call @loopCapture()
    %0: dynamic = const.undefined
    ret
}
