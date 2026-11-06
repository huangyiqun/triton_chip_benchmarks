/* Shared device-timing, device-query and JSON reporting for the Ascend C chip
 * benchmark suite.
 *
 * Timing policy mirrors triton_chip_benchmarks/common.py: a batch of launches
 * is bracketed by two ACL events, the elapsed time is divided by the actual
 * launch count, and the peak is the SHORTEST batch average observed while the
 * median is reported from the same configuration. Nothing is wall-clock timed
 * on the host, and compilation/allocation/validation never enter a sample.
 */
#ifndef ASCENDC_CHIP_BENCH_COMMON_H
#define ASCENDC_CHIP_BENCH_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"

/* ---------------------------------------------------------------- utilities */

#define BENCH_CHECK(expr)                                                              \
    do {                                                                               \
        aclError bench_err_ = (expr);                                                   \
        if (bench_err_ != ACL_SUCCESS) {                                                \
            fprintf(stderr, "[FATAL] %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__,       \
                    #expr, (int)bench_err_, aclGetRecentErrMsg());                      \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

static inline double bench_fabs(double v) { return v < 0 ? -v : v; }

static inline int bench_cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Median of a sample array (array is copied, caller's order is preserved). */
static inline double bench_median(const double* samples, int count) {
    if (count <= 0) return 0.0;
    double* tmp = (double*)malloc((size_t)count * sizeof(double));
    memcpy(tmp, samples, (size_t)count * sizeof(double));
    qsort(tmp, (size_t)count, sizeof(double), bench_cmp_double);
    double median = (count % 2) ? tmp[count / 2]
                                : 0.5 * (tmp[count / 2 - 1] + tmp[count / 2]);
    free(tmp);
    return median;
}

/* ------------------------------------------------------------ device probing */

typedef struct {
    int32_t device_id;
    char soc_name[128];
    int64_t aicore_num;      /* AI (Cube) cores   */
    int64_t cube_core_num;
    int64_t vector_core_num; /* AIV cores         */
    int64_t ub_per_vector_core;
    int64_t total_memory_bytes;
    int64_t l2_cache_bytes;
    int64_t npu_arch;
    double  aicore_freq_mhz; /* 0 when unavailable; read from npu-smi */
    size_t  free_memory_bytes;
} bench_device_info;

/* Read "Aicore Freq(MHZ)" from npu-smi. Returns 0.0 when unavailable; the
 * frequency is reported for context only and never scales a measurement. */
static inline double bench_read_aicore_freq_mhz(int32_t device_id) {
    char command[160];
    snprintf(command, sizeof(command),
             "npu-smi info -t common -i %d 2>/dev/null | grep 'Aicore Freq' | "
             "head -1 | tr -dc '0-9\\n'", (int)device_id);
    FILE* pipe = popen(command, "r");
    if (!pipe) return 0.0;
    char line[64] = {0};
    double value = 0.0;
    if (fgets(line, sizeof(line), pipe)) value = atof(line);
    pclose(pipe);
    return value;
}

static inline void bench_query_device(int32_t device_id, bench_device_info* info) {
    memset(info, 0, sizeof(*info));
    info->device_id = device_id;
    const char* soc = aclrtGetSocName();
    snprintf(info->soc_name, sizeof(info->soc_name), "%s", soc ? soc : "unknown");
    /* An unavailable attribute stays at its zeroed value; callers that need it
     * (for example the compute grid) must check rather than assume. */
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &info->aicore_num);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_CUBE_CORE_NUM, &info->cube_core_num);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &info->vector_core_num);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &info->ub_per_vector_core);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, &info->total_memory_bytes);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_L2_CACHE_SIZE, &info->l2_cache_bytes);
    aclrtGetDeviceInfo((uint32_t)device_id, ACL_DEV_ATTR_NPU_ARCH, &info->npu_arch);
    info->aicore_freq_mhz = bench_read_aicore_freq_mhz(device_id);
    size_t free_bytes = 0, total_bytes = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) == ACL_SUCCESS)
        info->free_memory_bytes = free_bytes;
}

/* -------------------------------------------------------------- event timing */

typedef struct {
    double batch_ms_target; /* aim for this much work per timed batch */
    double warmup_ms;       /* untimed warmup budget                 */
    double rep_ms;          /* total measurement budget              */
    int    rounds;          /* minimum number of timed batches       */
} bench_timing_config;

static inline bench_timing_config bench_default_timing(void) {
    bench_timing_config config;
    config.batch_ms_target = 10.0;
    config.warmup_ms = 100.0;
    config.rep_ms = 200.0;
    config.rounds = 10;
    return config;
}

typedef struct {
    double min_ms;      /* peak statistic: shortest batch average   */
    double median_ms;   /* sustained statistic, same configuration  */
    double max_ms;
    int    launches_per_batch;
    int    sample_count;
} bench_timing_result;

#define BENCH_MAX_LAUNCHES 4096
#define BENCH_MAX_SAMPLES  512

/* Time `launch(context)` with ACL events.
 *
 * `launch` must enqueue exactly one kernel launch on `stream` and must not
 * synchronize. The batch size is calibrated upward until the event timer
 * resolves it, so the reported per-launch time never divides by a quantized
 * zero. Returns 0 on success, -1 if the timer could not resolve any batch. */
static inline int bench_measure(void (*launch)(void*), void* context,
                                aclrtStream stream,
                                const bench_timing_config* config,
                                bench_timing_result* out) {
    aclrtEvent start = NULL, end = NULL;
    BENCH_CHECK(aclrtCreateEvent(&start));
    BENCH_CHECK(aclrtCreateEvent(&end));

    /* Compile/first-launch cost is excluded by launching once up front. */
    launch(context);
    BENCH_CHECK(aclrtSynchronizeStream(stream));

    int launches = 4;
    double batch_ms = 0.0;
    for (;;) {
        BENCH_CHECK(aclrtRecordEvent(start, stream));
        for (int i = 0; i < launches; ++i) launch(context);
        BENCH_CHECK(aclrtRecordEvent(end, stream));
        BENCH_CHECK(aclrtSynchronizeStream(stream));
        float elapsed = 0.0f;
        BENCH_CHECK(aclrtEventElapsedTime(&elapsed, start, end));
        batch_ms = (double)elapsed;
        /* Require a batch long enough that timer granularity cannot dominate. */
        if (batch_ms >= 0.5 && batch_ms >= config->batch_ms_target * 0.5) break;
        if (launches >= BENCH_MAX_LAUNCHES) {
            if (batch_ms <= 0.0) {
                aclrtDestroyEvent(start);
                aclrtDestroyEvent(end);
                return -1;
            }
            break;
        }
        int scaled = launches * 2;
        if (batch_ms > 0.0) {
            double wanted = launches * config->batch_ms_target / batch_ms;
            if (wanted > scaled) scaled = (int)(wanted + 0.5);
        }
        if (scaled > BENCH_MAX_LAUNCHES) scaled = BENCH_MAX_LAUNCHES;
        launches = scaled;
    }

    /* Untimed warmup at the final batch size. */
    int warmup_batches = (int)(config->warmup_ms / batch_ms) + 1;
    for (int w = 0; w < warmup_batches; ++w)
        for (int i = 0; i < launches; ++i) launch(context);
    BENCH_CHECK(aclrtSynchronizeStream(stream));

    int rounds = config->rounds;
    int budget_rounds = (int)(config->rep_ms / batch_ms) + 1;
    if (budget_rounds > rounds) rounds = budget_rounds;
    if (rounds > BENCH_MAX_SAMPLES) rounds = BENCH_MAX_SAMPLES;

    double samples[BENCH_MAX_SAMPLES];
    int collected = 0;
    for (int r = 0; r < rounds; ++r) {
        BENCH_CHECK(aclrtRecordEvent(start, stream));
        for (int i = 0; i < launches; ++i) launch(context);
        BENCH_CHECK(aclrtRecordEvent(end, stream));
        BENCH_CHECK(aclrtSynchronizeStream(stream));
        float elapsed = 0.0f;
        BENCH_CHECK(aclrtEventElapsedTime(&elapsed, start, end));
        if (elapsed <= 0.0f) continue;
        samples[collected++] = (double)elapsed / launches;
    }
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    if (collected == 0) return -1;

    out->min_ms = samples[0];
    out->max_ms = samples[0];
    for (int i = 1; i < collected; ++i) {
        if (samples[i] < out->min_ms) out->min_ms = samples[i];
        if (samples[i] > out->max_ms) out->max_ms = samples[i];
    }
    out->median_ms = bench_median(samples, collected);
    out->launches_per_batch = launches;
    out->sample_count = collected;
    return 0;
}

/* ------------------------------------------------------------- JSON emission */

/* Minimal streaming JSON writer. Comma placement is tracked per depth level so
 * callers never have to emit separators by hand. */
typedef struct {
    FILE* out;
    int   depth;
    int   needs_comma[32];
} bench_json;

static inline void bench_json_init(bench_json* json, FILE* out) {
    json->out = out;
    json->depth = 0;
    memset(json->needs_comma, 0, sizeof(json->needs_comma));
}

static inline void bench_json_separator(bench_json* json) {
    if (json->needs_comma[json->depth]) fputs(",", json->out);
    fputs("\n", json->out);
    for (int i = 0; i <= json->depth; ++i) fputs("  ", json->out);
    json->needs_comma[json->depth] = 1;
}

static inline void bench_json_open_object(bench_json* json, const char* key) {
    bench_json_separator(json);
    if (key) fprintf(json->out, "\"%s\": {", key); else fputs("{", json->out);
    json->depth++;
    json->needs_comma[json->depth] = 0;
}

static inline void bench_json_close_object(bench_json* json) {
    json->depth--;
    fputs("\n", json->out);
    for (int i = 0; i <= json->depth; ++i) fputs("  ", json->out);
    fputs("}", json->out);
    /* The closed object is itself a value at this level, so the next sibling
     * needs a separator. Callers that reset `depth` by hand must also set this,
     * which is what bench_json_return_to_root does. */
    json->needs_comma[json->depth] = 1;
}

static inline void bench_json_open_array(bench_json* json, const char* key) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": [", key);
    json->depth++;
    json->needs_comma[json->depth] = 0;
}

static inline void bench_json_close_array(bench_json* json) {
    json->depth--;
    fputs("\n", json->out);
    for (int i = 0; i <= json->depth; ++i) fputs("  ", json->out);
    fputs("]", json->out);
    json->needs_comma[json->depth] = 1;
}

/* Open and close the document's root object.
 *
 * The root is written here rather than by each caller so its comma state is
 * tracked like any other level: a hand-written "{" with a manually assigned
 * depth either emits a leading comma or drops the separator before the next
 * key, and both produce unparsable JSON. */
static inline void bench_json_begin_document(bench_json* json) {
    fputs("{", json->out);
    json->depth = 1;
    json->needs_comma[1] = 0;
}

static inline void bench_json_end_document(bench_json* json) {
    fputs("\n}\n", json->out);
    json->depth = 0;
}

/* Return to the root level after writing nested values. The nested writers
 * leave `depth` where they found it, so this only has to assert the level. */
static inline void bench_json_return_to_root(bench_json* json) {
    json->depth = 1;
}

static inline void bench_json_string(bench_json* json, const char* key, const char* value) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": \"%s\"", key, value ? value : "");
}

static inline void bench_json_double(bench_json* json, const char* key, double value) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": %.10g", key, value);
}

static inline void bench_json_int(bench_json* json, const char* key, long long value) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": %lld", key, value);
}

static inline void bench_json_bool(bench_json* json, const char* key, int value) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": %s", key, value ? "true" : "false");
}

/* A missing value is explicitly null; it is never replaced by zero. */
static inline void bench_json_null(bench_json* json, const char* key) {
    bench_json_separator(json);
    fprintf(json->out, "\"%s\": null", key);
}

static inline void bench_json_device(bench_json* json, const bench_device_info* info) {
    bench_json_open_object(json, "device");
    bench_json_int(json, "device_id", info->device_id);
    bench_json_string(json, "soc_name", info->soc_name);
    bench_json_int(json, "npu_arch", info->npu_arch);
    bench_json_int(json, "aicore_count", info->aicore_num);
    bench_json_int(json, "cube_core_count", info->cube_core_num);
    bench_json_int(json, "vector_core_count", info->vector_core_num);
    bench_json_int(json, "ub_bytes_per_vector_core", info->ub_per_vector_core);
    bench_json_int(json, "total_memory_bytes", info->total_memory_bytes);
    bench_json_double(json, "memory_capacity_gib",
                      (double)info->total_memory_bytes / 1073741824.0);
    bench_json_int(json, "l2_cache_bytes", info->l2_cache_bytes);
    if (info->aicore_freq_mhz > 0.0)
        bench_json_double(json, "aicore_freq_mhz", info->aicore_freq_mhz);
    else
        bench_json_null(json, "aicore_freq_mhz");
    bench_json_int(json, "free_memory_bytes", (long long)info->free_memory_bytes);
    bench_json_string(json, "query_source", "aclrtGetDeviceInfo / aclrtGetSocName");
    bench_json_close_object(json);
}

static inline void bench_json_timing(bench_json* json, const bench_timing_result* timing) {
    bench_json_open_object(json, "timing");
    bench_json_double(json, "min_ms", timing->min_ms);
    bench_json_double(json, "median_ms", timing->median_ms);
    bench_json_double(json, "max_ms", timing->max_ms);
    bench_json_int(json, "launches_per_batch", timing->launches_per_batch);
    bench_json_int(json, "sample_count", timing->sample_count);
    bench_json_string(json, "provider", "aclrtRecordEvent / aclrtEventElapsedTime");
    bench_json_string(json, "peak_statistic", "minimum sampled batch-average device time");
    bench_json_string(json, "sustained_statistic", "median sampled batch-average device time");
    bench_json_close_object(json);
}

/* ------------------------------------------------- half / bfloat16 host casts */

static inline uint16_t bench_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Round-to-nearest-even on the discarded 16 low bits. */
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static inline float bench_bf16_to_float(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t bench_float_to_fp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return (uint16_t)(sign << 15); /* subnormals flush to zero */
    if (exponent >= 31) return (uint16_t)((sign << 15) | 0x7C00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static inline float bench_fp16_to_float(uint16_t value) {
    uint32_t sign = (value >> 15) & 1u;
    uint32_t exponent = (value >> 10) & 0x1Fu;
    uint32_t mantissa = value & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) { bits = sign << 31; }
        else {
            /* Normalize the subnormal. */
            int shift = 0;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; ++shift; }
            mantissa &= 0x3FFu;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - shift + 1) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* FP8 E4M3FN: 1-4-3, no infinities, 0xFF/0x7F are NaN. */
static inline uint8_t bench_float_to_fp8_e4m3(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0 && mantissa == 0) return (uint8_t)(sign << 7);
    int32_t biased = exponent + 7;
    if (biased >= 15) return (uint8_t)((sign << 7) | 0x7Eu); /* clamp below NaN */
    if (biased <= 0) return (uint8_t)(sign << 7);            /* flush subnormals */
    return (uint8_t)((sign << 7) | ((uint32_t)biased << 3) | (mantissa >> 20));
}

static inline float bench_fp8_e4m3_to_float(uint8_t value) {
    uint32_t sign = (value >> 7) & 1u;
    uint32_t exponent = (value >> 3) & 0xFu;
    uint32_t mantissa = value & 0x7u;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        /* E4M3FN subnormal: 2^-6 * (mantissa / 8) */
        float magnitude = (float)mantissa / 8.0f * 0.015625f;
        return sign ? -magnitude : magnitude;
    }
    uint32_t bits = (sign << 31) | ((exponent - 7 + 127) << 23) | (mantissa << 20);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* FP4 E2M1: 1-2-1. Representable magnitudes are
 * {0, 0.5, 1, 1.5, 2, 3, 4, 6}; two values are packed per byte, low nibble
 * first. */
static inline uint8_t bench_float_to_fp4_e2m1(float value) {
    uint32_t sign = value < 0 ? 1u : 0u;
    float magnitude = value < 0 ? -value : value;
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    uint32_t code = 0;
    float best = 1e30f;
    for (uint32_t i = 0; i < 8; ++i) {
        float difference = magnitude - levels[i];
        if (difference < 0) difference = -difference;
        if (difference < best) { best = difference; code = i; }
    }
    return (uint8_t)((sign << 3) | code);
}

static inline float bench_fp4_e2m1_to_float(uint8_t nibble) {
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float magnitude = levels[nibble & 0x7u];
    return (nibble & 0x8u) ? -magnitude : magnitude;
}

#endif /* ASCENDC_CHIP_BENCH_COMMON_H */
