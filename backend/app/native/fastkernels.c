/*
 * fastkernels.c — низкоуровневые ядра для поиска по векторам и хэширования текста.
 *
 * Собирается автоматически (app/native/loader.py):
 *   gcc -O3 -march=native -ffast-math -funroll-loops -shared -fPIC fastkernels.c -o libfastidp.so
 * Если компилятор недоступен, Python-часть использует NumPy-реализацию с тем же результатом.
 *
 * Состав:
 *   1. Бинарное квантование 1 бит (упаковка в uint64) и расстояние Хэмминга через POPCNT.
 *   2. Скалярное квантование int8 (единичный вектор, масштаб 127) и скалярное произведение (AVX2).
 *   3. Хэширование текста fast_hashing_embedding: та же математика, что ml_pipeline.text_utils.hashing_embedding
 *      (токены [а-яё a-z 0-9]{3,}, признаки «первые 5 / последние 4 символа», MD5 mod dim), побитово совместимо.
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define FK_EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------------ */
/* Диагностика                                                          */
/* ------------------------------------------------------------------ */

FK_EXPORT int fk_has_avx2(void) {
#if defined(__AVX2__)
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

FK_EXPORT int fk_has_popcnt(void) {
    return __builtin_cpu_supports("popcnt") ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* 1. 1-битные сигнатуры и Хэмминг                                       */
/* ------------------------------------------------------------------ */

/* Порог — среднее значение вектора. Бит = 1, если компонента выше среднего.
 * words = ceil(dim / 64): для dim=1024 это 16 uint64 = 128 байт. */
FK_EXPORT void fk_pack_bits(const float *v, int64_t dim, uint64_t *out) {
    int64_t words = (dim + 63) / 64;
    double sum = 0.0;
    for (int64_t i = 0; i < dim; i++) sum += v[i];
    float mean = (float)(sum / (double)(dim > 0 ? dim : 1));
    memset(out, 0, (size_t)words * sizeof(uint64_t));
    for (int64_t i = 0; i < dim; i++) {
        if (v[i] > mean) out[i >> 6] |= (uint64_t)1 << (i & 63);
    }
}

/* SimHash: бит j = sign((v · R_j)), где R — dim×bits случайная матрица (row-major: строка i — коэффициенты
 * i-й компоненты для всех гиперплоскостей). Вероятность расхождения битов двух векторов равна θ/π,
 * поэтому расстояние Хэмминга оценивает угол. В отличие от знака самих координат, это хорошо работает
 * на разреженных векторах: проекция идёт только по ненулевым компонентам (cost = nnz × bits).
 * bits ≤ 4096, out — ceil(bits/64) слов. */
#define FK_MAX_BITS 4096
FK_EXPORT void fk_simhash_bits(const float *v, int64_t dim, const float *R, int64_t bits, uint64_t *out) {
    float acc[FK_MAX_BITS];
    int64_t words = (bits + 63) / 64;
    if (bits > FK_MAX_BITS) bits = FK_MAX_BITS;
    for (int64_t j = 0; j < bits; j++) acc[j] = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        float vi = v[i];
        if (vi == 0.0f) continue;
        const float *row = R + i * bits;
        for (int64_t j = 0; j < bits; j++) acc[j] += vi * row[j];
    }
    memset(out, 0, (size_t)words * sizeof(uint64_t));
    for (int64_t j = 0; j < bits; j++) {
        if (acc[j] > 0.0f) out[j >> 6] |= (uint64_t)1 << (j & 63);
    }
}

/* Расстояние Хэмминга между двумя сигнатурами (POPCNT). */
static inline int32_t hamming_words(const uint64_t *a, const uint64_t *b, int64_t words) {
    int32_t d = 0;
    for (int64_t w = 0; w < words; w++) {
        d += __builtin_popcountll(a[w] ^ b[w]);
    }
    return d;
}

/* Полный скан: dist[i] = hamming(sigs[i], q). sigs — n строк по words uint64. */
FK_EXPORT void fk_hamming_scan(const uint64_t *sigs, int64_t n, int64_t words,
                               const uint64_t *q, int32_t *dist) {
    for (int64_t i = 0; i < n; i++) {
        dist[i] = hamming_words(sigs + i * words, q, words);
    }
}

/* Top-k наименьших расстояний без выделения массива на n элементов.
 * Блоками по 8192 считаем расстояния и держим max-heap размером k.
 * Возвращает число найденных (min(k, n)); out_idx/out_dist отсортированы по возрастанию расстояния. */
#define FK_BLOCK 8192

static void heap_sift_down(int32_t *hd, int64_t *hi, int64_t size, int64_t pos) {
    for (;;) {
        int64_t l = 2 * pos + 1, r = l + 1, big = pos;
        if (l < size && (hd[l] > hd[big] || (hd[l] == hd[big] && hi[l] > hi[big]))) big = l;
        if (r < size && (hd[r] > hd[big] || (hd[r] == hd[big] && hi[r] > hi[big]))) big = r;
        if (big == pos) break;
        int32_t td = hd[pos]; hd[pos] = hd[big]; hd[big] = td;
        int64_t ti = hi[pos]; hi[pos] = hi[big]; hi[big] = ti;
        pos = big;
    }
}

FK_EXPORT int64_t fk_topk_hamming(const uint64_t *sigs, int64_t n, int64_t words,
                                  const uint64_t *q, int64_t k,
                                  int64_t *out_idx, int32_t *out_dist) {
    if (k <= 0 || n <= 0) return 0;
    int32_t *hd = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    int64_t *hi = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    int32_t *block = (int32_t *)malloc(FK_BLOCK * sizeof(int32_t));
    if (!hd || !hi || !block) {
        free(hd); free(hi); free(block);
        return -1;
    }
    int64_t size = 0;
    for (int64_t base = 0; base < n; base += FK_BLOCK) {
        int64_t cnt = n - base < FK_BLOCK ? n - base : FK_BLOCK;
        for (int64_t j = 0; j < cnt; j++) {
            block[j] = hamming_words(sigs + (base + j) * words, q, words);
        }
        for (int64_t j = 0; j < cnt; j++) {
            int32_t d = block[j];
            int64_t idx = base + j;
            if (size < k) {
                hd[size] = d; hi[size] = idx; size++;
                if (size == k) { /* построить кучу */
                    for (int64_t p = size / 2 - 1; p >= 0; p--) heap_sift_down(hd, hi, size, p);
                }
            } else if (d < hd[0] || (d == hd[0] && idx < hi[0])) {
                hd[0] = d; hi[0] = idx;
                heap_sift_down(hd, hi, size, 0);
            }
        }
    }
    /* Сортировка результата по возрастанию (distance, idx): heap-sort на той же max-куче, O(k log k).
     * (Раньше здесь была сортировка вставками O(k²): при IVF она вызывается nprobe раз на запрос.) */
    for (int64_t end = size - 1; end > 0; end--) {
        int32_t td = hd[0]; hd[0] = hd[end]; hd[end] = td;
        int64_t ti = hi[0]; hi[0] = hi[end]; hi[end] = ti;
        heap_sift_down(hd, hi, end, 0);
    }
    for (int64_t i = 0; i < size; i++) {
        out_idx[i] = hi[i];
        out_dist[i] = hd[i];
    }
    free(hd); free(hi); free(block);
    return size;
}

/* Top-k с ранним выходом (cascade / Matryoshka-подобный). Строки сканируются по одной:
 * сначала считаются первые early_words слов сигнатуры. Если порог кучи T уже известен (куча полна),
 * ожидаемое частичное расстояние кандидата при расстоянии T равно early_bits·T/bits. Если частичное
 * расстояние превышает это ожидание на z стандартных отклонений биномиального распределения, полное
 * расстояние почти наверняка не попадёт в top-k, и остальные слова не читаются.
 * Это СТАТИСТИЧЕСКИЙ отсев: изредка теряется кандидат, который был бы в top-k (контролируется z и замером recall).
 * Точное поведение — fk_topk_hamming (early_words = 0). Соседние строки предвыбираются (__builtin_prefetch). */
FK_EXPORT int64_t fk_topk_hamming_ee(const uint64_t *sigs, int64_t n, int64_t words, const uint64_t *q,
                                     int64_t k, int64_t early_words, double z,
                                     int64_t *out_idx, int32_t *out_dist) {
    if (k <= 0 || n <= 0) return 0;
    if (early_words <= 0 || early_words >= words) early_words = 0;
    int64_t bits = words * 64;
    int64_t early_bits = early_words * 64;
    int32_t *hd = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    int64_t *hi = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    if (!hd || !hi) { free(hd); free(hi); return -1; }
    int64_t size = 0;
    for (int64_t i = 0; i < n; i++) {
        if (i + 8 < n) __builtin_prefetch(sigs + (i + 8) * words, 0, 1);
        const uint64_t *row = sigs + i * words;
        if (early_words && size == k) {
            double T = (double)hd[0];
            double p = T / (double)bits;
            double lim = (double)early_bits * p + z * sqrt((double)early_bits * p * (1.0 - p));
            int32_t part = 0;
            for (int64_t w = 0; w < early_words; w++) part += __builtin_popcountll(row[w] ^ q[w]);
            if ((double)part > lim) continue;
        }
        int32_t d = hamming_words(row, q, words);
        if (size < k) {
            hd[size] = d; hi[size] = i; size++;
            if (size == k) {
                for (int64_t pos = size / 2 - 1; pos >= 0; pos--) heap_sift_down(hd, hi, size, pos);
            }
        } else if (d < hd[0] || (d == hd[0] && i < hi[0])) {
            hd[0] = d; hi[0] = i;
            heap_sift_down(hd, hi, size, 0);
        }
    }
    for (int64_t end = size - 1; end > 0; end--) {
        int32_t td = hd[0]; hd[0] = hd[end]; hd[end] = td;
        int64_t ti = hi[0]; hi[0] = hi[end]; hi[end] = ti;
        heap_sift_down(hd, hi, end, 0);
    }
    for (int64_t i = 0; i < size; i++) {
        out_idx[i] = hi[i];
        out_dist[i] = hd[i];
    }
    free(hd); free(hi);
    return size;
}

/* Top-k по нескольким диапазонам строк за один проход и с ОДНОЙ кучей (IVF: выбранные кластеры).
 * ranges — пары [lo, hi) в строках, nr — их число. Общая куча означает общий порог: он быстро ужесточается,
 * и ранний выход (early_words, z — см. fk_topk_hamming_ee) отсеивает больше. Индексы — абсолютные строки. */
FK_EXPORT int64_t fk_topk_hamming_ranges(const uint64_t *sigs, int64_t words, const uint64_t *q,
                                         const int64_t *ranges, int64_t nr, int64_t k,
                                         int64_t early_words, double z,
                                         int64_t *out_idx, int32_t *out_dist) {
    if (k <= 0 || nr <= 0) return 0;
    if (early_words <= 0 || early_words >= words) early_words = 0;
    int64_t bits = words * 64;
    int64_t early_bits = early_words * 64;
    int32_t *hd = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    int64_t *hi = (int64_t *)malloc((size_t)k * sizeof(int64_t));
    if (!hd || !hi) { free(hd); free(hi); return -1; }
    int64_t size = 0;
    for (int64_t r = 0; r < nr; r++) {
        int64_t lo = ranges[2 * r], hi_row = ranges[2 * r + 1];
        for (int64_t i = lo; i < hi_row; i++) {
            if (i + 8 < hi_row) __builtin_prefetch(sigs + (i + 8) * words, 0, 1);
            const uint64_t *row = sigs + i * words;
            if (early_words && size == k) {
                double T = (double)hd[0];
                double p = T / (double)bits;
                double lim = (double)early_bits * p + z * sqrt((double)early_bits * p * (1.0 - p));
                int32_t part = 0;
                for (int64_t w = 0; w < early_words; w++) part += __builtin_popcountll(row[w] ^ q[w]);
                if ((double)part > lim) continue;
            }
            int32_t d = hamming_words(row, q, words);
            if (size < k) {
                hd[size] = d; hi[size] = i; size++;
                if (size == k) {
                    for (int64_t pos = size / 2 - 1; pos >= 0; pos--) heap_sift_down(hd, hi, size, pos);
                }
            } else if (d < hd[0] || (d == hd[0] && i < hi[0])) {
                hd[0] = d; hi[0] = i;
                heap_sift_down(hd, hi, size, 0);
            }
        }
    }
    for (int64_t end = size - 1; end > 0; end--) {
        int32_t td = hd[0]; hd[0] = hd[end]; hd[end] = td;
        int64_t ti = hi[0]; hi[0] = hi[end]; hi[end] = ti;
        heap_sift_down(hd, hi, end, 0);
    }
    for (int64_t i = 0; i < size; i++) {
        out_idx[i] = hi[i];
        out_dist[i] = hd[i];
    }
    free(hd); free(hi);
    return size;
}

/* ------------------------------------------------------------------ */
/* 2. int8 SQ8: квантование единичного вектора и скалярное произведение   */
/* ------------------------------------------------------------------ */

/* Единичный вектор с фиксированным масштабом 127: компоненты в [-1, 1] → [-127, 127].
 * Для единичных векторов скалярное произведение int8 / 127² ≈ косинус. */
FK_EXPORT void fk_quantize_unit_i8(const float *v, int64_t dim, int8_t *out) {
    double ss = 0.0;
    for (int64_t i = 0; i < dim; i++) ss += (double)v[i] * (double)v[i];
    double norm = sqrt(ss);
    if (norm == 0.0) {
        memset(out, 0, (size_t)dim);
        return;
    }
    double inv = 127.0 / norm;
    for (int64_t i = 0; i < dim; i++) {
        double x = (double)v[i] * inv;
        long r = lround(x);
        if (r > 127) r = 127;
        if (r < -127) r = -127;
        out[i] = (int8_t)r;
    }
}

static inline int32_t dot_i8_scalar(const int8_t *a, const int8_t *b, int64_t dim) {
    int32_t s = 0;
    for (int64_t i = 0; i < dim; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

FK_EXPORT int32_t fk_dot_i8(const int8_t *a, const int8_t *b, int64_t dim) {
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));
        __m256i wa = _mm256_cvtepi8_epi16(va);
        __m256i wb = _mm256_cvtepi8_epi16(vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wa, wb));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    int32_t s = _mm_cvtsi128_si32(s4);
    for (; i < dim; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#else
    return dot_i8_scalar(a, b, dim);
#endif
}

/* Доранжирование кандидатов: scores[j] = dot(base[idx[j]], q). base — n×dim int8. */
FK_EXPORT void fk_rerank_i8(const int8_t *base, int64_t dim, const int64_t *idx, int64_t m,
                            const int8_t *q, int32_t *scores) {
    for (int64_t j = 0; j < m; j++) {
        scores[j] = fk_dot_i8(base + idx[j] * dim, q, dim);
    }
}

/* ------------------------------------------------------------------ */
/* 3. fast_hashing_embedding: MD5 (как hashlib.md5), UTF-8, токены         */
/* ------------------------------------------------------------------ */

static const uint32_t MD5_K[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u, 0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u,
};

static const uint8_t MD5_S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

static inline uint32_t rotl32(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

/* MD5 одного блока (сообщение ≤ 55 байт). Результат — 16 байт дайджеста. */
static void md5_short(const uint8_t *msg, size_t len, uint8_t out[16]) {
    uint8_t blk[64];
    memset(blk, 0, sizeof(blk));
    memcpy(blk, msg, len);
    blk[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8u;
    for (int i = 0; i < 8; i++) blk[56 + i] = (uint8_t)(bits >> (8 * i));

    uint32_t M[16];
    for (int i = 0; i < 16; i++) {
        M[i] = (uint32_t)blk[4 * i] | ((uint32_t)blk[4 * i + 1] << 8) |
               ((uint32_t)blk[4 * i + 2] << 16) | ((uint32_t)blk[4 * i + 3] << 24);
    }
    uint32_t A = 0x67452301u, B = 0xefcdab89u, C = 0x98badcfeu, D = 0x10325476u;
    uint32_t a0 = A, b0 = B, c0 = C, d0 = D;
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t F, g;
        if (i < 16) { F = (B & C) | (~B & D); g = i; }
        else if (i < 32) { F = (D & B) | (~D & C); g = (5 * i + 1) & 15; }
        else if (i < 48) { F = B ^ C ^ D; g = (3 * i + 5) & 15; }
        else { F = C ^ (B | ~D); g = (7 * i) & 15; }
        F = F + A + MD5_K[i] + M[g];
        A = D; D = C; C = B;
        B = B + rotl32(F, MD5_S[i]);
    }
    a0 += A; b0 += B; c0 += C; d0 += D;
    uint32_t words[4] = {a0, b0, c0, d0};
    for (int w = 0; w < 4; w++) {
        for (int b = 0; b < 4; b++) out[4 * w + b] = (uint8_t)(words[w] >> (8 * b));
    }
}

/* Ток. Python: [а-яё a-z 0-9]{3,} по text.lower(). Здесь — по кодовым точкам UTF-8. */
static inline int is_token_cp(uint32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
           (c >= 0x0430u && c <= 0x044Fu) || c == 0x0451u;
}

static inline uint32_t lower_cp(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32u;
    if (c >= 0x0410u && c <= 0x042Fu) return c + 32u;
    if (c == 0x0401u) return 0x0451u;
    return c;
}

/* Кодирование кодовой точки в UTF-8 (для наших символов не длиннее 2 байт). */
static inline size_t encode_cp(uint32_t c, uint8_t *dst) {
    if (c < 0x80u) { dst[0] = (uint8_t)c; return 1; }
    if (c < 0x800u) { dst[0] = (uint8_t)(0xC0u | (c >> 6)); dst[1] = (uint8_t)(0x80u | (c & 0x3Fu)); return 2; }
    dst[0] = (uint8_t)(0xE0u | (c >> 12));
    dst[1] = (uint8_t)(0x80u | ((c >> 6) & 0x3Fu));
    dst[2] = (uint8_t)(0x80u | (c & 0x3Fu));
    return 3;
}

/* Методы хэширования признаков. MD5 — совместим с Python (ml_pipeline.text_utils), используется по умолчанию.
 * WYHASH — быстрый некриптографический хэш + знаковое хэширование CountSketch (±1); эмбеддинги НЕ совместимы
 * с MD5, поэтому переключение требует переиндексации (см. docs). */
#define FK_HASH_MD5 0
#define FK_HASH_WYHASH 1

/* wyhash-подобный 64-битный хэш: умножение 64×64→128 бит (MUL), XOR-свёртка. Для коротких строк — 3–4 MUL. */
static inline uint64_t fk_mix(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

static uint64_t fk_wyhash(const uint8_t *p, size_t len, uint64_t seed) {
    const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull, s2 = 0x8ebc6af09c88c6e3ull;
    uint64_t h = seed ^ s0;
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = fk_mix(w ^ s1, h ^ s2);
        i += 8;
    }
    uint64_t w = 0;
    if (i < len) memcpy(&w, p + i, len - i);
    h = fk_mix(w ^ s1 ^ (uint64_t)len, h ^ s2);
    return fk_mix(h ^ s0, s1 ^ (uint64_t)len);
}

/* Добавляет один признак. MD5: дайджест mod dim, как в Python. WYHASH: индекс = h mod dim, знак = бит CountSketch. */
static void add_feature(int method, float *vec, int32_t dim, const uint32_t *cps, size_t n, double weight) {
    uint8_t msg[64];
    size_t len = 0;
    for (size_t i = 0; i < n; i++) len += encode_cp(cps[i], msg + len);
    if (method == FK_HASH_WYHASH) {
        uint64_t h = fk_wyhash(msg, len, 0x9E3779B97F4A7C15ull);
        uint64_t sign_bit = fk_mix(h, 0xD6E8FEB86659FD93ull) >> 63;
        float w = (float)weight;
        vec[h % (uint64_t)dim] += sign_bit ? -w : w;
        return;
    }
    uint8_t dig[16];
    md5_short(msg, len, dig);
    /* дайджест — 128-битное big-endian число: value = hi·2^64 + lo; r = value mod dim.
     * Два 64-битных остатка вместо 16 делений — тот же результат, что int(hexdigest, 16) % dim. */
    uint64_t hi = 0, lo = 0;
    for (int i = 0; i < 8; i++) hi = (hi << 8) | dig[i];
    for (int i = 8; i < 16; i++) lo = (lo << 8) | dig[i];
    uint64_t d = (uint64_t)dim;
    uint64_t two64_mod = (uint64_t)(((__uint128_t)1 << 64) % d);
    uint64_t r = ((hi % d) * two64_mod % d + lo % d) % d;
    vec[r] += (float)weight;
}

/* Хэш-эмбеддинг текста длиной len байт UTF-8 в out[dim] (нормированный L2).
 * Возвращает число токенов, либо -1 при ошибке. */
static int64_t hash_embed_core(const uint8_t *utf8, int64_t len, int32_t dim, float *out, int method) {
    if (dim <= 0) return -1;
    memset(out, 0, (size_t)dim * sizeof(float));

    uint32_t first[5];
    uint32_t ring[4];
    size_t tlen = 0;       /* длина текущего токена в кодовых точках */
    int64_t tokens = 0;

    /* Сброс: токен длиной ≥3 превращается в два признака. */
    #define FLUSH_TOKEN()                                                              \
        do {                                                                           \
            if (tlen >= 3) {                                                           \
                double weight = 1.0 / (1.0 + log(1.0 + (double)tlen));                 \
                uint32_t f1[5];                                                        \
                size_t n1 = tlen < 5 ? tlen : 5;                                       \
                memcpy(f1, first, n1 * sizeof(uint32_t));                              \
                add_feature(method, out, dim, f1, n1, weight);                                 \
                if (tlen > 4) {                                                        \
                    uint32_t f2[4];                                                    \
                    for (int j = 0; j < 4; j++) f2[j] = ring[(tlen - 4 + j) & 3];      \
                    add_feature(method, out, dim, f2, 4, weight);                              \
                } else {                                                               \
                    add_feature(method, out, dim, first, tlen, weight);                        \
                }                                                                      \
                tokens++;                                                              \
            }                                                                          \
            tlen = 0;                                                                  \
        } while (0)

    int64_t i = 0;
    while (i < len) {
        uint32_t c;
        uint8_t b0 = utf8[i];
        if (b0 < 0x80u) { c = b0; i += 1; }
        else if ((b0 & 0xE0u) == 0xC0u && i + 1 < len) {
            c = ((uint32_t)(b0 & 0x1Fu) << 6) | (utf8[i + 1] & 0x3Fu); i += 2;
        } else if ((b0 & 0xF0u) == 0xE0u && i + 2 < len) {
            c = ((uint32_t)(b0 & 0x0Fu) << 12) | ((uint32_t)(utf8[i + 1] & 0x3Fu) << 6) | (utf8[i + 2] & 0x3Fu);
            i += 3;
        } else { /* прочие байты (4-байтовые, битые) — разделители */
            c = 0; i += 1;
        }
        c = lower_cp(c);
        if (is_token_cp(c)) {
            if (tlen < 5) first[tlen] = c;
            ring[tlen & 3] = c;
            tlen++;
        } else {
            FLUSH_TOKEN();
        }
    }
    FLUSH_TOKEN();
    #undef FLUSH_TOKEN

    double ss = 0.0;
    for (int32_t k = 0; k < dim; k++) ss += (double)out[k] * (double)out[k];
    if (ss > 0.0) {
        float inv = (float)(1.0 / sqrt(ss));
        for (int32_t k = 0; k < dim; k++) out[k] *= inv;
    }
    return tokens;
}

FK_EXPORT int64_t fk_hash_embed(const uint8_t *utf8, int64_t len, int32_t dim, float *out) {
    return hash_embed_core(utf8, len, dim, out, FK_HASH_MD5);
}

/* ------------------------------------------------------------------ */
/* 5. Пакетные функции с OpenMP (без GIL: один вызов из Python на весь батч) */
/* ------------------------------------------------------------------ */

FK_EXPORT int32_t fk_has_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/* Батч хэш-эмбеддингов. buf — все тексты подряд (UTF-8), offs[n+1] — границы; out — n×dim float32.
 * method: 0 = MD5 (совместим с Python), 1 = wyhash + CountSketch. Потоки пишут в разные строки out. */
FK_EXPORT void fk_hash_embed_batch(const uint8_t *buf, const int64_t *offs, int64_t n, int32_t dim,
                                   int32_t method, float *out) {
    #pragma omp parallel for schedule(guided)
    for (int64_t i = 0; i < n; i++) {
        hash_embed_core(buf + offs[i], offs[i + 1] - offs[i], dim, out + (size_t)i * (size_t)dim, method);
    }
}

/* Быстрое преобразование Уолша–Адамара без нормировки, in-place, n — степень двойки.
 * Только сложения и вычитания: O(n log n), без умножений. Внутренний цикл векторизуется компилятором. */
FK_EXPORT void fk_fwht(float *x, int64_t n) {
    for (int64_t h = 1; h < n; h <<= 1) {
        for (int64_t i = 0; i < n; i += 2 * h) {
            for (int64_t j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

/* SimHash с ортогональным вращением: y = H·D·v, где D — диагональ случайных знаков (dsign, длина pad),
 * H — преобразование Адамара (pad — степень двойки ≥ dim). Бит j = [y_j > 0] для j < bits.
 * Стоимость O(pad log pad) вместо O(dim²) у плотной матрицы проекции. */
static void simhash_fwht_one(const float *v, int64_t dim, const float *dsign, int64_t pad, int64_t bits,
                             float *buf, uint64_t *out) {
    for (int64_t i = 0; i < pad; i++) buf[i] = (i < dim) ? v[i] * dsign[i] : 0.0f;
    fk_fwht(buf, pad);
    int64_t words = (bits + 63) / 64;
    memset(out, 0, (size_t)words * sizeof(uint64_t));
    for (int64_t j = 0; j < bits; j++) {
        if (buf[j] > 0.0f) out[j >> 6] |= (uint64_t)1 << (j & 63);
    }
}

FK_EXPORT void fk_simhash_fwht(const float *v, int64_t dim, const float *dsign, int64_t pad, int64_t bits,
                               uint64_t *out) {
    float *buf = (float *)malloc((size_t)pad * sizeof(float));
    if (!buf) return;
    simhash_fwht_one(v, dim, dsign, pad, bits, buf, out);
    free(buf);
}

/* Батч SimHash: V — n×dim float32, out — n×words uint64. */
FK_EXPORT void fk_simhash_fwht_batch(const float *V, int64_t n, int64_t dim, const float *dsign, int64_t pad,
                                     int64_t bits, uint64_t *out) {
    int64_t words = (bits + 63) / 64;
    #pragma omp parallel
    {
        float *buf = (float *)malloc((size_t)pad * sizeof(float));
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            if (buf) simhash_fwht_one(V + (size_t)i * (size_t)dim, dim, dsign, pad, bits, buf,
                                      out + (size_t)i * (size_t)words);
        }
        free(buf);
    }
}

/* Батч квантования int8: V — n×dim float32, out — n×dim int8 (единичные векторы, масштаб 127). */
FK_EXPORT void fk_quantize_unit_i8_batch(const float *V, int64_t n, int64_t dim, int8_t *out) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        fk_quantize_unit_i8(V + (size_t)i * (size_t)dim, dim, out + (size_t)i * (size_t)dim);
    }
}
