module D:/projects/brass/tests/bronze_corpus/01_arithmetic.js

func mathTest(%0: f64, %1: f64) -> f64 {
  b0:
    %2: f64 = add %0, %1
    %3: f64 = sub %0, %1
    %4: f64 = mul %0, %1
    %5: f64 = div %0, %1
    %6: f64 = mod %0, %1
    %7: f64 = neg %0
    %8: f64 = add %2, %3
    %9: f64 = add %8, %4
    %10: f64 = add %9, %5
    %11: f64 = add %10, %6
    %12: f64 = add %11, %7
    ret %12
}

func run() -> void {
  b0:
    %0: f64 = const.f64 100
    %1: f64 = const.f64 25
    %2: f64 = call @mathTest(%0, %1)
    %3: f64 = const.f64 7
    %4: f64 = const.f64 3
    %5: f64 = call @mathTest(%3, %4)
    %6: dynamic = name.resolve "print"
    %7: dynamic = const.undefined
    %8: dynamic = box.f64 %2
    %9: dynamic = box.f64 %5
    %10: dynamic = call.dynamic %6, %7, 2, %8, %9
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
