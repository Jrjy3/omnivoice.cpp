// Model-free GGUF metadata type-safety tests.
#include "gguf-weights.h"

#include <cstdio>
#include <exception>

static void probe_u32(const GGUFModel & gf) { (void) gf_get_u32(gf, "wrong.u32"); }
static void probe_f32(const GGUFModel & gf) { (void) gf_get_f32(gf, "wrong.f32"); }
static void probe_str(const GGUFModel & gf) { (void) gf_get_str(gf, "wrong.str"); }
static void probe_bool(const GGUFModel & gf) { (void) gf_get_bool(gf, "wrong.bool"); }

int main() {
    GGUFModel gf = {};
    gf.gguf = gguf_init_empty();
    if (!gf.gguf) {
        return 1;
    }

    gguf_set_val_i32(gf.gguf, "wrong.u32", 7);
    gguf_set_val_u32(gf.gguf, "wrong.f32", 7);
    gguf_set_val_bool(gf.gguf, "wrong.str", true);
    gguf_set_val_str(gf.gguf, "wrong.bool", "true");

    struct ScalarCase {
        const char * name;
        void (*probe)(const GGUFModel &);
    } scalar_cases[] = {
        { "u32", probe_u32 },
        { "f32", probe_f32 },
        { "str", probe_str },
        { "bool", probe_bool },
    };
    for (const ScalarCase & test : scalar_cases) {
        bool threw = false;
        try {
            test.probe(gf);
        } catch (const std::exception &) {
            threw = true;
        }
        if (!threw) {
            std::fprintf(stderr, "[GGUF-metadata] wrong %s type did not fail cleanly\n", test.name);
            gguf_free(gf.gguf);
            return 2;
        }
    }

    gguf_set_val_u32(gf.gguf, "good.u32", 42);
    gguf_set_val_f32(gf.gguf, "good.f32", 0.25f);
    gguf_set_val_str(gf.gguf, "good.str", "value");
    gguf_set_val_bool(gf.gguf, "good.bool", true);
    if (gf_get_u32(gf, "good.u32") != 42 || gf_get_f32(gf, "good.f32") != 0.25f ||
        std::strcmp(gf_get_str(gf, "good.str"), "value") != 0 || !gf_get_bool(gf, "good.bool") ||
        gf_get_u32(gf, "missing") != 0) {
        std::fprintf(stderr, "[GGUF-metadata] valid or missing scalar behavior changed\n");
        gguf_free(gf.gguf);
        return 3;
    }

    const int expected[] = { 2, 5 };
    const uint32_t good_u32[] = { 2, 5 };
    const int32_t good_i32[] = { 2, 5 };
    const float wrong_type[] = { 2.0f, 5.0f };
    const uint32_t wrong_length[] = { 2 };
    const uint32_t wrong_value[] = { 2, 6 };
    gguf_set_arr_data(gf.gguf, "array.good_u32", GGUF_TYPE_UINT32, good_u32, 2);
    gguf_set_arr_data(gf.gguf, "array.good_i32", GGUF_TYPE_INT32, good_i32, 2);
    gguf_set_val_u32(gf.gguf, "array.scalar", 2);
    gguf_set_arr_data(gf.gguf, "array.wrong_type", GGUF_TYPE_FLOAT32, wrong_type, 2);
    gguf_set_arr_data(gf.gguf, "array.wrong_length", GGUF_TYPE_UINT32, wrong_length, 1);
    gguf_set_arr_data(gf.gguf, "array.wrong_value", GGUF_TYPE_UINT32, wrong_value, 2);

    if (!gf_array_equals_i32(gf, "array.good_u32", expected, 2) ||
        !gf_array_equals_i32(gf, "array.good_i32", expected, 2) ||
        gf_array_equals_i32(gf, "array.scalar", expected, 2) ||
        gf_array_equals_i32(gf, "array.wrong_type", expected, 2) ||
        gf_array_equals_i32(gf, "array.wrong_length", expected, 2) ||
        gf_array_equals_i32(gf, "array.wrong_value", expected, 2)) {
        std::fprintf(stderr, "[GGUF-metadata] array type/length/value validation failed\n");
        gguf_free(gf.gguf);
        return 4;
    }

    gguf_free(gf.gguf);
    return 0;
}
