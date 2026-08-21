module D:/projects/brass/tests/bronze_corpus/05_fib_rec.js

func fibRec(%0: f64) -> f64 {
  b0:
    %1: f64 = const.f64 2
    %2: bool = cmp.lt %0, %1
    br %2, b1, b2
  b1:
    ret %0
  b2:
    jump b3
  b3:
    %3: f64 = const.f64 1
    %4: f64 = sub %0, %3
    %5: f64 = call @fibRec(%4)
    %6: f64 = const.f64 2
    %7: f64 = sub %0, %6
    %8: f64 = call @fibRec(%7)
    %9: f64 = add %5, %8
    ret %9
}

func run() -> void {
  b0:
    %0: f64 = const.f64 5
    %1: f64 = call @fibRec(%0)
    %2: f64 = const.f64 10
    %3: f64 = call @fibRec(%2)
    %4: f64 = const.f64 15
    %5: f64 = call @fibRec(%4)
    %6: f64 = const.f64 34
    %7: f64 = call @fibRec(%6)
    %8: dynamic = name.resolve "print"
    %9: dynamic = const.undefined
    %10: dynamic = box.f64 %1
    %11: dynamic = box.f64 %3
    %12: dynamic = box.f64 %5
    %13: dynamic = box.f64 %7
    %14: dynamic = call.dynamic %8, %9, 4, %10, %11, %12, %13
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
