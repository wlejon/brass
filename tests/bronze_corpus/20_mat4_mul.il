module D:/projects/brass/tests/bronze_corpus/20_mat4_mul.js

func Matrix4() -> dynamic {
  b0:
    %0: dynamic = create.array 0
    %1: f64 = const.f64 0
    jump b1(%1)
  b1(%2: f64):
    %3: f64 = const.f64 16
    %4: bool = cmp.lt %2, %3
    br %4, b2, b3(%2)
  b2:
    %6: dynamic = box.f64 %2
    %7: f64 = const.f64 5
    %8: f64 = mod %2, %7
    %9: f64 = const.f64 0
    %10: bool = cmp.eq %8, %9
    br %10, b4, b5
  b3(%5: f64):
    %17: dynamic = create.object
    prop.set %17, 6, %0, 0, 0
    ret %17
  b4:
    %11: f64 = const.f64 1
    jump b6(%11)
  b5:
    %12: f64 = const.f64 0
    jump b6(%12)
  b6(%13: f64):
    %14: dynamic = box.f64 %13
    elem.set %0, %6, %14, 0
    %15: f64 = const.f64 1
    %16: f64 = add %2, %15
    jump b1(%16)
}

func getElem(%0: dynamic, %1: f64) -> dynamic {
  b0:
    %2: dynamic = prop.get %0, 6, 1, mono
    %3: dynamic = box.f64 %1
    %4: dynamic = elem.get %2, %3
    ret %4
}

func setElem(%0: dynamic, %1: f64, %2: dynamic) -> void {
  b0:
    %3: dynamic = prop.get %0, 6, 2, mono
    %4: dynamic = box.f64 %1
    elem.set %3, %4, %2, 0
    ret
}

func multiplyMatrices(%0: dynamic, %1: dynamic, %2: dynamic) -> dynamic {
  b0:
    %3: dynamic = prop.get %1, 6, 3, mono
    %4: dynamic = prop.get %2, 6, 4, mono
    %5: dynamic = prop.get %0, 6, 5, mono
    %6: f64 = const.f64 0
    %7: f64 = const.f64 0
    %8: f64 = add %6, %7
    %9: dynamic = box.f64 %8
    %10: dynamic = elem.get %3, %9
    %11: f64 = const.f64 0
    %12: f64 = const.f64 4
    %13: f64 = add %11, %12
    %14: dynamic = box.f64 %13
    %15: dynamic = elem.get %3, %14
    %16: f64 = const.f64 0
    %17: f64 = const.f64 8
    %18: f64 = add %16, %17
    %19: dynamic = box.f64 %18
    %20: dynamic = elem.get %3, %19
    %21: f64 = const.f64 0
    %22: f64 = const.f64 12
    %23: f64 = add %21, %22
    %24: dynamic = box.f64 %23
    %25: dynamic = elem.get %3, %24
    %26: f64 = const.f64 0
    %27: f64 = const.f64 1
    %28: f64 = add %26, %27
    %29: dynamic = box.f64 %28
    %30: dynamic = elem.get %3, %29
    %31: f64 = const.f64 0
    %32: f64 = const.f64 5
    %33: f64 = add %31, %32
    %34: dynamic = box.f64 %33
    %35: dynamic = elem.get %3, %34
    %36: f64 = const.f64 0
    %37: f64 = const.f64 9
    %38: f64 = add %36, %37
    %39: dynamic = box.f64 %38
    %40: dynamic = elem.get %3, %39
    %41: f64 = const.f64 0
    %42: f64 = const.f64 13
    %43: f64 = add %41, %42
    %44: dynamic = box.f64 %43
    %45: dynamic = elem.get %3, %44
    %46: f64 = const.f64 0
    %47: f64 = const.f64 2
    %48: f64 = add %46, %47
    %49: dynamic = box.f64 %48
    %50: dynamic = elem.get %3, %49
    %51: f64 = const.f64 0
    %52: f64 = const.f64 6
    %53: f64 = add %51, %52
    %54: dynamic = box.f64 %53
    %55: dynamic = elem.get %3, %54
    %56: f64 = const.f64 0
    %57: f64 = const.f64 10
    %58: f64 = add %56, %57
    %59: dynamic = box.f64 %58
    %60: dynamic = elem.get %3, %59
    %61: f64 = const.f64 0
    %62: f64 = const.f64 14
    %63: f64 = add %61, %62
    %64: dynamic = box.f64 %63
    %65: dynamic = elem.get %3, %64
    %66: f64 = const.f64 0
    %67: f64 = const.f64 3
    %68: f64 = add %66, %67
    %69: dynamic = box.f64 %68
    %70: dynamic = elem.get %3, %69
    %71: f64 = const.f64 0
    %72: f64 = const.f64 7
    %73: f64 = add %71, %72
    %74: dynamic = box.f64 %73
    %75: dynamic = elem.get %3, %74
    %76: f64 = const.f64 0
    %77: f64 = const.f64 11
    %78: f64 = add %76, %77
    %79: dynamic = box.f64 %78
    %80: dynamic = elem.get %3, %79
    %81: f64 = const.f64 0
    %82: f64 = const.f64 15
    %83: f64 = add %81, %82
    %84: dynamic = box.f64 %83
    %85: dynamic = elem.get %3, %84
    %86: f64 = const.f64 0
    %87: f64 = const.f64 0
    %88: f64 = add %86, %87
    %89: dynamic = box.f64 %88
    %90: dynamic = elem.get %4, %89
    %91: f64 = const.f64 0
    %92: f64 = const.f64 4
    %93: f64 = add %91, %92
    %94: dynamic = box.f64 %93
    %95: dynamic = elem.get %4, %94
    %96: f64 = const.f64 0
    %97: f64 = const.f64 8
    %98: f64 = add %96, %97
    %99: dynamic = box.f64 %98
    %100: dynamic = elem.get %4, %99
    %101: f64 = const.f64 0
    %102: f64 = const.f64 12
    %103: f64 = add %101, %102
    %104: dynamic = box.f64 %103
    %105: dynamic = elem.get %4, %104
    %106: f64 = const.f64 0
    %107: f64 = const.f64 1
    %108: f64 = add %106, %107
    %109: dynamic = box.f64 %108
    %110: dynamic = elem.get %4, %109
    %111: f64 = const.f64 0
    %112: f64 = const.f64 5
    %113: f64 = add %111, %112
    %114: dynamic = box.f64 %113
    %115: dynamic = elem.get %4, %114
    %116: f64 = const.f64 0
    %117: f64 = const.f64 9
    %118: f64 = add %116, %117
    %119: dynamic = box.f64 %118
    %120: dynamic = elem.get %4, %119
    %121: f64 = const.f64 0
    %122: f64 = const.f64 13
    %123: f64 = add %121, %122
    %124: dynamic = box.f64 %123
    %125: dynamic = elem.get %4, %124
    %126: f64 = const.f64 0
    %127: f64 = const.f64 2
    %128: f64 = add %126, %127
    %129: dynamic = box.f64 %128
    %130: dynamic = elem.get %4, %129
    %131: f64 = const.f64 0
    %132: f64 = const.f64 6
    %133: f64 = add %131, %132
    %134: dynamic = box.f64 %133
    %135: dynamic = elem.get %4, %134
    %136: f64 = const.f64 0
    %137: f64 = const.f64 10
    %138: f64 = add %136, %137
    %139: dynamic = box.f64 %138
    %140: dynamic = elem.get %4, %139
    %141: f64 = const.f64 0
    %142: f64 = const.f64 14
    %143: f64 = add %141, %142
    %144: dynamic = box.f64 %143
    %145: dynamic = elem.get %4, %144
    %146: f64 = const.f64 0
    %147: f64 = const.f64 3
    %148: f64 = add %146, %147
    %149: dynamic = box.f64 %148
    %150: dynamic = elem.get %4, %149
    %151: f64 = const.f64 0
    %152: f64 = const.f64 7
    %153: f64 = add %151, %152
    %154: dynamic = box.f64 %153
    %155: dynamic = elem.get %4, %154
    %156: f64 = const.f64 0
    %157: f64 = const.f64 11
    %158: f64 = add %156, %157
    %159: dynamic = box.f64 %158
    %160: dynamic = elem.get %4, %159
    %161: f64 = const.f64 0
    %162: f64 = const.f64 15
    %163: f64 = add %161, %162
    %164: dynamic = box.f64 %163
    %165: dynamic = elem.get %4, %164
    %166: f64 = const.f64 0
    %167: f64 = const.f64 0
    %168: f64 = add %166, %167
    %169: dynamic = box.f64 %168
    %170: dynamic = mul %10, %90
    %171: dynamic = mul %15, %110
    %172: dynamic = add %170, %171
    %173: dynamic = mul %20, %130
    %174: dynamic = add %172, %173
    %175: dynamic = mul %25, %150
    %176: dynamic = add %174, %175
    elem.set %5, %169, %176, 0
    %177: f64 = const.f64 0
    %178: f64 = const.f64 4
    %179: f64 = add %177, %178
    %180: dynamic = box.f64 %179
    %181: dynamic = mul %10, %95
    %182: dynamic = mul %15, %115
    %183: dynamic = add %181, %182
    %184: dynamic = mul %20, %135
    %185: dynamic = add %183, %184
    %186: dynamic = mul %25, %155
    %187: dynamic = add %185, %186
    elem.set %5, %180, %187, 0
    %188: f64 = const.f64 0
    %189: f64 = const.f64 8
    %190: f64 = add %188, %189
    %191: dynamic = box.f64 %190
    %192: dynamic = mul %10, %100
    %193: dynamic = mul %15, %120
    %194: dynamic = add %192, %193
    %195: dynamic = mul %20, %140
    %196: dynamic = add %194, %195
    %197: dynamic = mul %25, %160
    %198: dynamic = add %196, %197
    elem.set %5, %191, %198, 0
    %199: f64 = const.f64 0
    %200: f64 = const.f64 12
    %201: f64 = add %199, %200
    %202: dynamic = box.f64 %201
    %203: dynamic = mul %10, %105
    %204: dynamic = mul %15, %125
    %205: dynamic = add %203, %204
    %206: dynamic = mul %20, %145
    %207: dynamic = add %205, %206
    %208: dynamic = mul %25, %165
    %209: dynamic = add %207, %208
    elem.set %5, %202, %209, 0
    %210: f64 = const.f64 0
    %211: f64 = const.f64 1
    %212: f64 = add %210, %211
    %213: dynamic = box.f64 %212
    %214: dynamic = mul %30, %90
    %215: dynamic = mul %35, %110
    %216: dynamic = add %214, %215
    %217: dynamic = mul %40, %130
    %218: dynamic = add %216, %217
    %219: dynamic = mul %45, %150
    %220: dynamic = add %218, %219
    elem.set %5, %213, %220, 0
    %221: f64 = const.f64 0
    %222: f64 = const.f64 5
    %223: f64 = add %221, %222
    %224: dynamic = box.f64 %223
    %225: dynamic = mul %30, %95
    %226: dynamic = mul %35, %115
    %227: dynamic = add %225, %226
    %228: dynamic = mul %40, %135
    %229: dynamic = add %227, %228
    %230: dynamic = mul %45, %155
    %231: dynamic = add %229, %230
    elem.set %5, %224, %231, 0
    %232: f64 = const.f64 0
    %233: f64 = const.f64 9
    %234: f64 = add %232, %233
    %235: dynamic = box.f64 %234
    %236: dynamic = mul %30, %100
    %237: dynamic = mul %35, %120
    %238: dynamic = add %236, %237
    %239: dynamic = mul %40, %140
    %240: dynamic = add %238, %239
    %241: dynamic = mul %45, %160
    %242: dynamic = add %240, %241
    elem.set %5, %235, %242, 0
    %243: f64 = const.f64 0
    %244: f64 = const.f64 13
    %245: f64 = add %243, %244
    %246: dynamic = box.f64 %245
    %247: dynamic = mul %30, %105
    %248: dynamic = mul %35, %125
    %249: dynamic = add %247, %248
    %250: dynamic = mul %40, %145
    %251: dynamic = add %249, %250
    %252: dynamic = mul %45, %165
    %253: dynamic = add %251, %252
    elem.set %5, %246, %253, 0
    %254: f64 = const.f64 0
    %255: f64 = const.f64 2
    %256: f64 = add %254, %255
    %257: dynamic = box.f64 %256
    %258: dynamic = mul %50, %90
    %259: dynamic = mul %55, %110
    %260: dynamic = add %258, %259
    %261: dynamic = mul %60, %130
    %262: dynamic = add %260, %261
    %263: dynamic = mul %65, %150
    %264: dynamic = add %262, %263
    elem.set %5, %257, %264, 0
    %265: f64 = const.f64 0
    %266: f64 = const.f64 6
    %267: f64 = add %265, %266
    %268: dynamic = box.f64 %267
    %269: dynamic = mul %50, %95
    %270: dynamic = mul %55, %115
    %271: dynamic = add %269, %270
    %272: dynamic = mul %60, %135
    %273: dynamic = add %271, %272
    %274: dynamic = mul %65, %155
    %275: dynamic = add %273, %274
    elem.set %5, %268, %275, 0
    %276: f64 = const.f64 0
    %277: f64 = const.f64 10
    %278: f64 = add %276, %277
    %279: dynamic = box.f64 %278
    %280: dynamic = mul %50, %100
    %281: dynamic = mul %55, %120
    %282: dynamic = add %280, %281
    %283: dynamic = mul %60, %140
    %284: dynamic = add %282, %283
    %285: dynamic = mul %65, %160
    %286: dynamic = add %284, %285
    elem.set %5, %279, %286, 0
    %287: f64 = const.f64 0
    %288: f64 = const.f64 14
    %289: f64 = add %287, %288
    %290: dynamic = box.f64 %289
    %291: dynamic = mul %50, %105
    %292: dynamic = mul %55, %125
    %293: dynamic = add %291, %292
    %294: dynamic = mul %60, %145
    %295: dynamic = add %293, %294
    %296: dynamic = mul %65, %165
    %297: dynamic = add %295, %296
    elem.set %5, %290, %297, 0
    %298: f64 = const.f64 0
    %299: f64 = const.f64 3
    %300: f64 = add %298, %299
    %301: dynamic = box.f64 %300
    %302: dynamic = mul %70, %90
    %303: dynamic = mul %75, %110
    %304: dynamic = add %302, %303
    %305: dynamic = mul %80, %130
    %306: dynamic = add %304, %305
    %307: dynamic = mul %85, %150
    %308: dynamic = add %306, %307
    elem.set %5, %301, %308, 0
    %309: f64 = const.f64 0
    %310: f64 = const.f64 7
    %311: f64 = add %309, %310
    %312: dynamic = box.f64 %311
    %313: dynamic = mul %70, %95
    %314: dynamic = mul %75, %115
    %315: dynamic = add %313, %314
    %316: dynamic = mul %80, %135
    %317: dynamic = add %315, %316
    %318: dynamic = mul %85, %155
    %319: dynamic = add %317, %318
    elem.set %5, %312, %319, 0
    %320: f64 = const.f64 0
    %321: f64 = const.f64 11
    %322: f64 = add %320, %321
    %323: dynamic = box.f64 %322
    %324: dynamic = mul %70, %100
    %325: dynamic = mul %75, %120
    %326: dynamic = add %324, %325
    %327: dynamic = mul %80, %140
    %328: dynamic = add %326, %327
    %329: dynamic = mul %85, %160
    %330: dynamic = add %328, %329
    elem.set %5, %323, %330, 0
    %331: f64 = const.f64 0
    %332: f64 = const.f64 15
    %333: f64 = add %331, %332
    %334: dynamic = box.f64 %333
    %335: dynamic = mul %70, %105
    %336: dynamic = mul %75, %125
    %337: dynamic = add %335, %336
    %338: dynamic = mul %80, %145
    %339: dynamic = add %337, %338
    %340: dynamic = mul %85, %165
    %341: dynamic = add %339, %340
    elem.set %5, %334, %341, 0
    ret %0
}

func testMat4Mul(%0: f64) -> dynamic {
  b0:
    %1: dynamic = call @Matrix4()
    %2: dynamic = call @Matrix4()
    %3: dynamic = call @Matrix4()
    %4: f64 = const.f64 0
    %5: f64 = const.f64 0.9999995
    %6: dynamic = box.f64 %5
    call @setElem(%2, %4, %6)
    %7: dynamic = const.undefined
    %8: f64 = const.f64 4
    %9: f64 = const.f64 0.0009999998
    %10: f64 = neg %9
    %11: dynamic = box.f64 %10
    call @setElem(%2, %8, %11)
    %12: dynamic = const.undefined
    %13: f64 = const.f64 1
    %14: f64 = const.f64 0.0009999998
    %15: dynamic = box.f64 %14
    call @setElem(%2, %13, %15)
    %16: dynamic = const.undefined
    %17: f64 = const.f64 5
    %18: f64 = const.f64 0.9999995
    %19: dynamic = box.f64 %18
    call @setElem(%2, %17, %19)
    %20: dynamic = const.undefined
    %21: f64 = const.f64 12
    %22: f64 = const.f64 1e-04
    %23: dynamic = box.f64 %22
    call @setElem(%2, %21, %23)
    %24: dynamic = const.undefined
    %25: f64 = const.f64 13
    %26: f64 = const.f64 2e-04
    %27: dynamic = box.f64 %26
    call @setElem(%2, %25, %27)
    %28: dynamic = const.undefined
    %29: f64 = const.f64 0
    jump b1(%29)
  b1(%30: f64):
    %31: bool = cmp.lt %30, %0
    br %31, b2, b3(%30)
  b2:
    %33: dynamic = call @multiplyMatrices(%3, %1, %2)
    %34: f64 = const.f64 0
    %35: f64 = const.f64 0
    %36: dynamic = call @getElem(%3, %35)
    call @setElem(%1, %34, %36)
    %37: dynamic = const.undefined
    %38: f64 = const.f64 4
    %39: f64 = const.f64 4
    %40: dynamic = call @getElem(%3, %39)
    call @setElem(%1, %38, %40)
    %41: dynamic = const.undefined
    %42: f64 = const.f64 1
    %43: f64 = const.f64 1
    %44: dynamic = call @getElem(%3, %43)
    call @setElem(%1, %42, %44)
    %45: dynamic = const.undefined
    %46: f64 = const.f64 5
    %47: f64 = const.f64 5
    %48: dynamic = call @getElem(%3, %47)
    call @setElem(%1, %46, %48)
    %49: dynamic = const.undefined
    %50: f64 = const.f64 12
    %51: f64 = const.f64 12
    %52: dynamic = call @getElem(%3, %51)
    %53: f64 = const.f64 0.99999
    %54: dynamic = box.f64 %53
    %55: dynamic = mul %52, %54
    call @setElem(%1, %50, %55)
    %56: dynamic = const.undefined
    %57: f64 = const.f64 13
    %58: f64 = const.f64 13
    %59: dynamic = call @getElem(%3, %58)
    %60: f64 = const.f64 0.99999
    %61: dynamic = box.f64 %60
    %62: dynamic = mul %59, %61
    call @setElem(%1, %57, %62)
    %63: dynamic = const.undefined
    %64: f64 = const.f64 1
    %65: f64 = add %30, %64
    jump b1(%65)
  b3(%32: f64):
    %66: f64 = const.f64 0
    %67: f64 = const.f64 0
    %70: dynamic = box.f64 %66
    jump b4(%70, %67)
  b4(%68: dynamic, %69: f64):
    %71: f64 = const.f64 16
    %72: bool = cmp.lt %69, %71
    br %72, b5, b6(%68, %69)
  b5:
    %75: dynamic = call @getElem(%3, %69)
    %76: dynamic = add %68, %75
    %77: f64 = const.f64 1
    %78: f64 = add %69, %77
    jump b4(%76, %78)
  b6(%73: dynamic, %74: f64):
    ret %73
}

func run() -> void {
  b0:
    %0: f64 = const.f64 4e+05
    %1: dynamic = call @testMat4Mul(%0)
    %2: dynamic = name.resolve "print"
    %3: dynamic = const.undefined
    %4: dynamic = call.dynamic %2, %3, 1, %1
    ret
}

func main() -> void {
  b0:
    call @run()
    %0: dynamic = const.undefined
    ret
}
