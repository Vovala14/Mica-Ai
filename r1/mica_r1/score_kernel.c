/* Exact fused scoring: old-field reads, int32 sums, first candidate wins ties.
 * No SIMD instruction-set requirement, floating point, or parallel runtime. */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
void mica_winners(int64_t rows, int64_t cells, int64_t channels,
                  int64_t pages, int64_t candidates, int64_t terms,
                  int64_t selectors, const int32_t *field,
                  const int64_t *batch, const int64_t *cell,
                  const int64_t *model, const int64_t *page,
                  const int32_t *bases, const int32_t *nb,
                  const int32_t *ch, const int32_t *co,
                  const int32_t *count, const int32_t *bias, int64_t *out) {
    for (int64_t r=0; r<rows; ++r) {
        const int32_t *f=field + batch[r]*cells*channels;
        const int32_t *base=bases + cell[r]*selectors;
        int64_t p=(model[r]*pages+page[r])*candidates;
        int32_t best=INT32_MIN;
        int64_t winner=0;
        for (int64_t j=0; j<candidates; ++j) {
            int32_t score=bias[p+j];
            int64_t t0=(p+j)*terms;
            for (int32_t t=0; t<count[p+j]; ++t) {
                int64_t x=t0+t;
                score += co[x]*f[base[nb[x]]+ch[x]];
            }
            if (score>best) { best=score; winner=j; }
        }
        out[r]=winner;
    }
}
