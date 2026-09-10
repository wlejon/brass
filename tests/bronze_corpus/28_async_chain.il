module tests/bronze_corpus/28_async_chain.js

func step1(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = env.get %0, 0, 0
    %2: f64 = unbox.f64 %1
    %3: f64 = const.f64 10
    %4: f64 = add %2, %3
    %5: dynamic = box.f64 %4
    ret %5
}

func step2(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = env.get %0, 0, 0
    %2: f64 = unbox.f64 %1
    %3: f64 = const.f64 2
    %4: f64 = mul %2, %3
    %5: dynamic = box.f64 %4
    ret %5
}

func step3(%0: dynamic) -> dynamic {
  b0:
    %1: dynamic = env.get %0, 0, 0
    %2: f64 = unbox.f64 %1
    %3: f64 = const.f64 5
    %4: f64 = add %2, %3
    %5: dynamic = box.f64 %4
    ret %5
}

func run() -> void {
  b0:
    %0: f64 = const.f64 5
    %1: dynamic = box.f64 %0

    %2: dynamic = create.async_machine @step1, 16, %1
    %3: dynamic = async.start %2, %1
    %4: dynamic = async.await %2, %3

    %5: dynamic = create.async_machine @step2, 16, %4
    %6: dynamic = async.start %5, %4
    %7: dynamic = async.await %5, %6

    %8: dynamic = create.async_machine @step3, 16, %7
    %9: dynamic = async.start %8, %7
    %10: dynamic = async.await %8, %9

    %11: f64 = unbox.f64 %10
    print %11
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
