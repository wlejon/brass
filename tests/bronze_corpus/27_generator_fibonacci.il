module tests/bronze_corpus/27_generator_fibonacci.js

func fibonacci$coro() -> dynamic {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = const.f64 1
    jump b1(%0, %1)
  b1(%2: f64, %3: f64):
    %4: dynamic = box.f64 %2
    %5: dynamic = yield %4
    %6: f64 = add %2, %3
    jump b1(%3, %6)
}

func fibonacci() -> dynamic {
  b0:
    %0: dynamic = const.undefined
    %1: dynamic = create.async_machine @fibonacci$coro, 16, %0
    ret %1
}

func run() -> void {
  b0:
    %0: dynamic = call @fibonacci()
    %1: dynamic = iter.open %0
    %2: f64 = const.f64 0
    %3: f64 = const.f64 0
    jump b1(%2, %3)
  b1(%4: f64, %5: f64):
    %6: f64 = const.f64 10
    %7: bool = cmp.lt %5, %6
    br %7, b2, b3(%4)
  b2:
    %8: dynamic = iter.step %1
    %9: dynamic = prop.get %8, "value"
    %10: f64 = unbox.f64 %9
    %11: f64 = add %4, %10
    %12: f64 = const.f64 1
    %13: f64 = add %5, %12
    jump b1(%11, %13)
  b3(%14: f64):
    print %14
    ret
}

func main() -> void {
  b0:
    call @run()
    ret
}
