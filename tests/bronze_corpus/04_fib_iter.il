module D:/projects/brass/tests/bronze_corpus/04_fib_iter.js

func fibIter(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 0
    %2: f64 = const.f64 1
    %3: f64 = const.f64 0
    jump b1(%1, %2, %3)
  b1(%4: f64, %5: f64, %6: f64):
    %7: bool = cmp.lt %6, %0
    br %7, b2, b3(%4, %5, %6)
  b2:
    %11: f64 = add %4, %5
    %12: f64 = const.f64 1000000007
    %13: f64 = mod %11, %12
    %14: f64 = const.f64 1
    %15: f64 = add %6, %14
    jump b1(%5, %13, %15)
  b3(%8: f64, %9: f64, %10: f64):
    ret %8
}

func run() -> void {
  b0:
    %0: f64 = const.f64 0
    %1: f64 = call @fibIter(%0)
    %2: f64 = const.f64 1
    %3: f64 = call @fibIter(%2)
    %4: f64 = const.f64 10
    %5: f64 = call @fibIter(%4)
    %6: f64 = const.f64 20
    %7: f64 = call @fibIter(%6)
    %8: f64 = const.f64 30
    %9: f64 = call @fibIter(%8)
    %10: f64 = const.f64 2.5e+07
    %11: f64 = call @fibIter(%10)
    %12: dynamic = name.resolve "print"
    %13: dynamic = const.undefined
    %14: dynamic = box.f64 %1
    %15: dynamic = box.f64 %3
    %16: dynamic = box.f64 %5
    %17: dynamic = box.f64 %7
    %18: dynamic = box.f64 %9
    %19: dynamic = box.f64 %11
    %20: dynamic = call.dynamic %12, %13, 6, %14, %15, %16, %17, %18, %19
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
