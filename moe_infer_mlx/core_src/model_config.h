// Runtime model configuration — loaded from model_config.json at init time.
// Types (ExpertLayout, ModelConfig) are in common.h via common.h.

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "common.h"

static int json_int(NSDictionary *d, NSString *key, int fallback) {
    NSNumber *v = d[key];
    return v ? [v intValue] : fallback;
}

static void parse_layout(ExpertLayout *l, NSDictionary *d) {
    l->gate_w_off = json_int(d, @"gate_w_off", 0);
    l->gate_s_off = json_int(d, @"gate_s_off", 0);
    l->gate_b_off = json_int(d, @"gate_b_off", 0);
    l->up_w_off   = json_int(d, @"up_w_off", 0);
    l->up_s_off   = json_int(d, @"up_s_off", 0);
    l->up_b_off   = json_int(d, @"up_b_off", 0);
    l->down_w_off = json_int(d, @"down_w_off", 0);
    l->down_s_off = json_int(d, @"down_s_off", 0);
    l->down_b_off = json_int(d, @"down_b_off", 0);
    l->gate_w_size = json_int(d, @"gate_w_size", 0);
    l->gate_s_size = json_int(d, @"gate_s_size", 0);
    l->gate_b_size = json_int(d, @"gate_b_size", 0);
    l->up_w_size   = json_int(d, @"up_w_size", 0);
    l->up_s_size   = json_int(d, @"up_s_size", 0);
    l->up_b_size   = json_int(d, @"up_b_size", 0);
    l->down_w_size = json_int(d, @"down_w_size", 0);
    l->down_s_size = json_int(d, @"down_s_size", 0);
    l->down_b_size = json_int(d, @"down_b_size", 0);
}

static int model_config_load(FlashMoE_Context *m) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", m->model_path);

    NSString *ns_path = [NSString stringWithUTF8String:path];
    NSData *data = [NSData dataWithContentsOfFile:ns_path];
    if (!data) {
        fprintf(stderr, "ERROR: Cannot read %s\n", path);
        return -1;
    }

    NSError *err = nil;
    NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data
                                                         options:0
                                                           error:&err];
    if (!json) {
        fprintf(stderr, "ERROR: Invalid JSON in %s: %s\n",
                path, [[err localizedDescription] UTF8String]);
        return -1;
    }

    m->cfg.hidden_dim      = json_int(json, @"hidden_dim", 2048);
    m->cfg.num_layers      = json_int(json, @"num_layers", 40);
    m->cfg.num_attn_heads  = json_int(json, @"num_attn_heads", 16);
    m->cfg.num_kv_heads    = json_int(json, @"num_kv_heads", 2);
    m->cfg.vocab_size      = json_int(json, @"vocab_size", 248320);
    m->cfg.num_experts     = json_int(json, @"num_experts", 256);
    m->cfg.num_experts_per_tok = json_int(json, @"num_experts_per_tok", 8);
    m->cfg.moe_intermediate     = json_int(json, @"moe_intermediate", 512);
    m->cfg.shared_intermediate  = json_int(json, @"shared_intermediate", 512);
    m->cfg.linear_num_v_heads   = json_int(json, @"linear_num_v_heads", 32);
    m->cfg.linear_num_k_heads   = json_int(json, @"linear_num_k_heads", 16);
    m->cfg.rotary_dim           = json_int(json, @"rotary_dim", 64);
    m->cfg.linear_total_key     = json_int(json, @"linear_total_key", 2048);
    m->cfg.linear_total_value   = json_int(json, @"linear_total_value", 4096);
    m->cfg.linear_conv_dim      = json_int(json, @"linear_conv_dim", 8192);
    m->cfg.num_full_attn_layers = json_int(json, @"num_full_attn_layers", 10);
    m->cfg.num_linear_layers    = json_int(json, @"num_linear_layers", 30);
    m->cfg.expert_size_4bit     = json_int(json, @"expert_size_4bit", 1769472);
    m->cfg.expert_size_2bit     = json_int(json, @"expert_size_2bit", 983040);

    NSDictionary *l4 = json[@"expert_layout_4bit"];
    if (l4) parse_layout(&m->cfg.layout_4bit, l4);
    NSDictionary *l2 = json[@"expert_layout_2bit"];
    if (l2) parse_layout(&m->cfg.layout_2bit, l2);

    // Architectural constants — read from JSON, fall back to Qwen3 defaults
    m->cfg.head_dim           = json_int(json, @"head_dim", 256);
    m->cfg.group_size         = json_int(json, @"group_size", 64);
    m->cfg.full_attn_interval = json_int(json, @"full_attn_interval", 4);
    m->cfg.conv_kernel_size   = json_int(json, @"conv_kernel_size", 4);
    m->cfg.max_seq_len        = json_int(json, @"max_seq_len", 1048576);
    m->cfg.gpu_kv_seq         = json_int(json, @"gpu_kv_seq", 8192);
    m->cfg.max_k              = json_int(json, @"max_k", 8);
    m->cfg.linear_key_dim     = json_int(json, @"linear_key_dim", 128);
    m->cfg.linear_value_dim   = json_int(json, @"linear_value_dim", 128);
    m->cfg.rms_norm_eps       = (float)(json[@"rms_norm_eps"] ? [json[@"rms_norm_eps"] doubleValue] : 1e-6);
    m->cfg.rope_theta         = (float)(json[@"rope_theta"] ? [json[@"rope_theta"] doubleValue] : 10000000.0);

    if (m->cfg.max_k > MAX_K) {
        fprintf(stderr, "ERROR: model max_k=%d exceeds compile-time MAX_K=%d\n",
                m->cfg.max_k, MAX_K);
        return -1;
    }

    printf("[config] hidden_dim=%d, num_layers=%d (%d full + %d linear)\n",
           m->cfg.hidden_dim, m->cfg.num_layers,
           m->cfg.num_full_attn_layers, m->cfg.num_linear_layers);
    printf("  experts=%d (K=%d), moe_inter=%d, shared_inter=%d\n",
           m->cfg.num_experts, m->cfg.num_experts_per_tok,
           m->cfg.moe_intermediate, m->cfg.shared_intermediate);
    printf("  attn_heads=%d, kv_heads=%d, vocab=%d, head_dim=%d\n",
           m->cfg.num_attn_heads, m->cfg.num_kv_heads,
           m->cfg.vocab_size, m->cfg.head_dim);
    printf("  linear: v_heads=%d, k_heads=%d\n",
           m->cfg.linear_num_v_heads, m->cfg.linear_num_k_heads);
    printf("  expert_size=%d bytes (4-bit), %d bytes (2-bit)\n",
           m->cfg.expert_size_4bit, m->cfg.expert_size_2bit);

    return 0;
}

#endif // MODEL_CONFIG_H
