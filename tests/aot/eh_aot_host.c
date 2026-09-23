/* Host for the AOT exception test: calls the brass functions compiled from
 * eh_aot.mir into a COFF object and prints one "<function> <arg> <result>"
 * line per call, for tests/cmake/run_aot_eh.cmake to check against the
 * interpreter. Every call with arg > 0 throws inside brass code. */
#include <stdint.h>
#include <stdio.h>

int64_t brass_main(int64_t);
int64_t main_resume(int64_t);
int64_t loop_eh(int64_t);
int64_t live_pad(int64_t);

int main(void) {
    static const struct {
        const char* name;
        int64_t (*fn)(int64_t);
    } fns[] = {
        {"brass_main", brass_main},
        {"main_resume", main_resume},
        {"loop_eh", loop_eh},
        {"live_pad", live_pad},
    };
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int64_t a = 0; a < 8; ++a) {
        for (size_t i = 0; i < sizeof(fns) / sizeof(fns[0]); ++i) {
            printf("%s %lld %lld\n", fns[i].name, (long long)a, (long long)fns[i].fn(a));
        }
    }
    printf("done\n");
    return 0;
}
