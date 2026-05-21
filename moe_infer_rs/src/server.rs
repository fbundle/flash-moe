/// HTTP server with OpenAI-compatible /v1/chat/completions (SSE streaming).
///
/// Usage: cargo run --release -- --serve 8000 --model data/
/// Clients send `{"model": "model-name", "messages": [...]}` — the server
/// loads model files from `data/model-name/`.
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{IntoRawFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use crate::config::{load_model_config, ModelConfig};
use crate::error::MoEError;
use crate::gpu_forward::{self, LinearAttnState};
use crate::metal_context::{metal_buf_shared, GpuWeightCtx, MetalContext};
use crate::quant::{bf16_to_f32, cpu_dequant_matvec_4bit, cpu_rms_norm};
use crate::tokenizer::BpeTokenizer;
use crate::weights::WeightFile;

const EOS_TOKEN_1: usize = 248046;
const EOS_TOKEN_2: usize = 248044;
const RMS_NORM_EPS: f32 = 1e-6;
const FULL_ATTN_INTERVAL: usize = 4;
const GROUP_SIZE: usize = 64;

const SSE_HEADERS: &str = "\
    HTTP/1.1 200 OK\r\n\
    Content-Type: text/event-stream\r\n\
    Cache-Control: no-cache\r\n\
    Connection: close\r\n\
    Access-Control-Allow-Origin: *\r\n\
    \r\n";

const CORS_RESPONSE: &str = "\
    HTTP/1.1 204 No Content\r\n\
    Access-Control-Allow-Origin: *\r\n\
    Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
    Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
    Access-Control-Max-Age: 86400\r\n\
    \r\n";

// ─── HTTP helpers ─────────────────────────────────────────────────────────

fn read_http_request(stream: &mut TcpStream) -> Result<Vec<u8>, std::io::Error> {
    stream.set_nonblocking(false)?;
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];

    loop {
        stream.read_exact(&mut byte)?;
        buf.push(byte[0]);
        let len = buf.len();
        if len >= 4
            && buf[len - 4] == b'\r'
            && buf[len - 3] == b'\n'
            && buf[len - 2] == b'\r'
            && buf[len - 1] == b'\n'
        {
            break;
        }
        if len > 1024 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "request too large",
            ));
        }
    }

    let header_str = String::from_utf8_lossy(&buf);
    let content_len = header_str
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().parse::<usize>().ok());

    if let Some(cl) = content_len {
        if cl > 0 {
            let mut body = vec![0u8; cl];
            stream.read_exact(&mut body)?;
            buf.extend_from_slice(&body);
        }
    }

    Ok(buf)
}

fn http_write_all(mut stream: &TcpStream, data: &[u8]) {
    let _ = stream.write_all(data);
}

fn http_write_str(stream: &TcpStream, s: &str) {
    http_write_all(stream, s.as_bytes());
}

fn sse_send_delta(mut stream: &TcpStream, request_id: &str, token_text: &str) -> bool {
    let escaped = token_text
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");

    let chunk = format!(
        "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
        request_id, escaped
    );
    stream.write(chunk.as_bytes()).unwrap_or(0) > 0
}

fn sse_send_done(mut stream: &TcpStream, request_id: &str) {
    let chunk = format!(
        "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
        request_id
    );
    let _ = stream.write(chunk.as_bytes());
}

// ─── Chat message formatting ──────────────────────────────────────────────

static DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant. /think";

fn tokenize_chat_messages(
    tokenizer: &BpeTokenizer,
    messages: &serde_json::Value,
) -> Result<Vec<usize>, MoEError> {
    let empty_vec = vec![];
    let msgs = messages.as_array().unwrap_or(&empty_vec);
    let mut prompt = String::new();

    // Always include system message first
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(DEFAULT_SYSTEM_PROMPT);
    prompt.push_str("<|im_end|>\n");

    for msg in msgs {
        let role = msg["role"].as_str().unwrap_or("user");
        let content = msg["content"].as_str().unwrap_or("");
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);
        prompt.push_str("<|im_end|>\n");
    }

    // Generation prompt (thinking mode)
    prompt.push_str("<|im_start|>assistant\n<think>\n");

    Ok(tokenizer
        .encode(&prompt, 8192)
        .into_iter()
        .map(|id| id as usize)
        .collect())
}

// ─── Embedding lookup ─────────────────────────────────────────────────────

fn embed_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
    let w_data = wf.get_tensor_u32("model.embed_tokens.weight");
    let s_data = wf.get_tensor_u16("model.embed_tokens.scales");
    let b_data = wf.get_tensor_u16("model.embed_tokens.biases");

    let (Some(w), Some(s), Some(b)) = (w_data, s_data, b_data) else {
        out.fill(0.0);
        return;
    };

    let w_info = wf.get_tensor_info("model.embed_tokens.weight").unwrap();
    let packed_cols = w_info.shape[1];
    let s_info = wf.get_tensor_info("model.embed_tokens.scales").unwrap();
    let num_groups = s_info.shape[1];
    let group_size = hidden_dim / num_groups;
    let packed_per_group = group_size / 8;

    let w_row = &w[token_id * packed_cols..];
    let s_row = &s[token_id * num_groups..];
    let b_row = &b[token_id * num_groups..];

    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        let base = g * group_size;

        for p in 0..packed_per_group {
            let packed = w_row[g * packed_per_group + p];
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + p * 8 + n] = (nibble as f32) * scale + bias;
            }
        }
    }
}

// ─── Layer state ──────────────────────────────────────────────────────────

struct KVCache {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    len: usize,
}

impl KVCache {
    fn new(max_len: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        KVCache {
            k_cache: vec![0.0f32; max_len * kv_dim],
            v_cache: vec![0.0f32; max_len * kv_dim],
            len: 0,
        }
    }
}

enum LayerState {
    FullAttention(KVCache),
    LinearAttention(LinearAttnState),
}

// ─── RoPE ─────────────────────────────────────────────────────────────────

fn apply_rotary_emb(
    q: &mut [f32], k: &mut [f32], pos: usize,
    num_q_heads: usize, num_kv_heads: usize,
    head_dim: usize, rotary_dim: usize,
    rope_theta: f64,
) {
    let theta_base = rope_theta;
    let pos_f = pos as f32;

    for h in 0..num_q_heads {
        let qh = &mut q[h * head_dim..];
        for d in (0..rotary_dim).step_by(2) {
            let theta = pos_f as f64 * theta_base.powf(-2.0 * (d as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (q0, q1) = (qh[d], qh[d + 1]);
            qh[d] = q0 * cos - q1 * sin;
            qh[d + 1] = q0 * sin + q1 * cos;
        }
    }

    for h in 0..num_kv_heads {
        let kh = &mut k[h * head_dim..];
        for d in (0..rotary_dim).step_by(2) {
            let theta = pos_f as f64 * theta_base.powf(-2.0 * (d as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (k0, k1) = (kh[d], kh[d + 1]);
            kh[d] = k0 * cos - k1 * sin;
            kh[d + 1] = k0 * sin + k1 * cos;
        }
    }
}

// ─── Full attention ───────────────────────────────────────────────────────

fn full_attention_forward(
    wf: &WeightFile, layer_idx: usize,
    hidden: &mut [f32], kv: &mut KVCache, pos: usize,
    hidden_dim: usize, num_attn_heads: usize, num_kv_heads: usize,
    head_dim: usize, rotary_dim: usize,
    rope_theta: f64,
    gpu_wf: Option<&GpuWeightCtx>,
    ctx: Option<&MetalContext>,
) {
    let q_proj_dim = num_attn_heads * head_dim * 2;
    let q_dim = num_attn_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        cpu_rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    let q_name = format!("model.layers.{}.self_attn.q_proj", layer_idx);
    let k_name = format!("model.layers.{}.self_attn.k_proj", layer_idx);
    let v_name = format!("model.layers.{}.self_attn.v_proj", layer_idx);

    // Q/K/V projections: GPU if available, else CPU
    let mut q_proj_out = vec![0.0f32; q_proj_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
        // GPU: dispatch Q, K, V in one encoder
        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim); }
        let qbuf = metal_buf_shared(&c.device, q_proj_dim * 4);
        let kbuf = metal_buf_shared(&c.device, kv_dim * 4);
        let vbuf = metal_buf_shared(&c.device, kv_dim * 4);

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, &format!("{}", q_name), &x_buf, 0, &qbuf, 0, q_proj_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}", k_name), &x_buf, 0, &kbuf, 0, kv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}", v_name), &x_buf, 0, &vbuf, 0, kv_dim, hidden_dim);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(qbuf.contents() as *const f32, q_proj_out.as_mut_ptr(), q_proj_dim);
            std::ptr::copy_nonoverlapping(kbuf.contents() as *const f32, k.as_mut_ptr(), kv_dim);
            std::ptr::copy_nonoverlapping(vbuf.contents() as *const f32, v.as_mut_ptr(), kv_dim);
        }
    } else {
        // CPU fallback
        if let (Some(qw), Some(qs), Some(qb)) = (
            wf.get_tensor_u32(&format!("{}.weight", q_name)),
            wf.get_tensor_u16(&format!("{}.scales", q_name)),
            wf.get_tensor_u16(&format!("{}.biases", q_name)),
        ) {
            cpu_dequant_matvec_4bit(qw, qs, qb, &normed, &mut q_proj_out, q_proj_dim, hidden_dim, GROUP_SIZE);
        }
        if let (Some(kw), Some(ks), Some(kb)) = (
            wf.get_tensor_u32(&format!("{}.weight", k_name)),
            wf.get_tensor_u16(&format!("{}.scales", k_name)),
            wf.get_tensor_u16(&format!("{}.biases", k_name)),
        ) {
            cpu_dequant_matvec_4bit(kw, ks, kb, &normed, &mut k, kv_dim, hidden_dim, GROUP_SIZE);
        }
        if let (Some(vw), Some(vs), Some(vb)) = (
            wf.get_tensor_u32(&format!("{}.weight", v_name)),
            wf.get_tensor_u16(&format!("{}.scales", v_name)),
            wf.get_tensor_u16(&format!("{}.biases", v_name)),
        ) {
            cpu_dequant_matvec_4bit(vw, vs, vb, &normed, &mut v, kv_dim, hidden_dim, GROUP_SIZE);
        }
    }

    let mut q = vec![0.0f32; q_dim];
    let mut q_gate = vec![0.0f32; q_dim];
    for h in 0..num_attn_heads {
        let src = &q_proj_out[h * 2 * head_dim..];
        q[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[..head_dim]);
        q_gate[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
    }

    if let Some(qnw) = wf.get_tensor_u16(&format!("model.layers.{}.self_attn.q_norm.weight", layer_idx)) {
        for h in 0..num_attn_heads {
            let qh = &mut q[h * head_dim..];
            let sum_sq: f32 = qh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            let n = qh.len().min(qnw.len());
            for i in 0..n { qh[i] = qh[i] * inv_rms * bf16_to_f32(qnw[i]); }
        }
    }
    if let Some(knw) = wf.get_tensor_u16(&format!("model.layers.{}.self_attn.k_norm.weight", layer_idx)) {
        for h in 0..num_kv_heads {
            let kh = &mut k[h * head_dim..];
            let sum_sq: f32 = kh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            let n = kh.len().min(knw.len());
            for i in 0..n { kh[i] = kh[i] * inv_rms * bf16_to_f32(knw[i]); }
        }
    }

    apply_rotary_emb(&mut q, &mut k, pos, num_attn_heads, num_kv_heads, head_dim, rotary_dim, rope_theta);

    let cache_pos = kv.len;
    let start = cache_pos * kv_dim;
    kv.k_cache[start..start + kv_dim].copy_from_slice(&k);
    kv.v_cache[start..start + kv_dim].copy_from_slice(&v);
    kv.len += 1;

    let heads_per_kv = num_attn_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; q_dim];

    for h in 0..num_attn_heads {
        let kv_h = h / heads_per_kv;
        let qh = &q[h * head_dim..];
        let seq_len = kv.len;

        let mut scores = vec![0.0f32; seq_len];
        for p in 0..seq_len {
            let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..];
            scores[p] = qh.iter().zip(kp.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
        }

        let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = scores.iter().map(|&s| (s - max_val).exp()).sum();
        let inv_sum = 1.0 / sum;

        let oh = &mut attn_out[h * head_dim..];
        for p in 0..seq_len {
            let weight = (scores[p] - max_val).exp() * inv_sum;
            let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..];
            for d in 0..head_dim { oh[d] += weight * vp[d]; }
        }
    }

    for i in 0..q_dim {
        attn_out[i] *= 1.0f32 / (1.0f32 + (-q_gate[i]).exp());
    }

    // O projection: GPU if available, else CPU
    let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
    let mut o_out = vec![0.0f32; hidden_dim];
    if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
        let attn_buf = metal_buf_shared(&c.device, q_dim * 4);
        unsafe { let dst = attn_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(attn_out.as_ptr(), dst, q_dim); }
        let out_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, &o_prefix, &attn_buf, 0, &out_buf, 0, hidden_dim, q_dim);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, o_out.as_mut_ptr(), hidden_dim); }
    } else if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.weight", o_prefix)),
        wf.get_tensor_u16(&format!("{}.scales", o_prefix)),
        wf.get_tensor_u16(&format!("{}.biases", o_prefix)),
    ) {
        cpu_dequant_matvec_4bit(ow, os, ob, &attn_out, &mut o_out, hidden_dim, q_dim, GROUP_SIZE);
    }

    for i in 0..hidden_dim { hidden[i] += o_out[i]; }
}

// ─── LM head ──────────────────────────────────────────────────────────────

fn lm_head_forward(
    wf: &WeightFile, hidden: &[f32], logits: &mut [f32],
    gpu_wf: Option<&GpuWeightCtx>, ctx: Option<&MetalContext>,
) {
    if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
        let x_buf = metal_buf_shared(&c.device, hidden.len() * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(hidden.as_ptr(), dst, hidden.len()); }
        let out_buf = metal_buf_shared(&c.device, logits.len() * 4);
        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len()); }
    } else {
        let w = wf.get_tensor_u32("lm_head.weight");
        let s = wf.get_tensor_u16("lm_head.scales");
        let b = wf.get_tensor_u16("lm_head.biases");
        let (Some(w_data), Some(s_data), Some(b_data)) = (w, s, b) else {
            logits[0] = 1.0;
            return;
        };
        cpu_dequant_matvec_4bit(w_data, s_data, b_data, hidden, logits, logits.len(), hidden.len(), GROUP_SIZE);
    }
}

fn cpu_argmax(x: &[f32]) -> usize {
    x.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

// ─── Debug helpers ──────────────────────────────────────────────────────────

fn stats_f32(x: &[f32]) -> (f32, f32, f32, f32, bool) {
    if x.is_empty() { return (0.0, 0.0, 0.0, 0.0, false); }
    let mut has_nan = false;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;
    let mut sq = 0.0;
    for &v in x {
        if v.is_nan() { has_nan = true; continue; }
        min = min.min(v);
        max = max.max(v);
        sum += v;
        sq += v * v;
    }
    let n = x.len() as f32;
    let mean = sum / n;
    let std = ((sq / n) - mean * mean).max(0.0).sqrt();
    (min, max, mean, std, has_nan)
}

// ─── Vocab ────────────────────────────────────────────────────────────────

struct Vocabulary {
    tokens: Vec<String>,
}

impl Vocabulary {
    fn from_tokenizer(tok: &BpeTokenizer) -> Self {
        let mut entries: Vec<(u32, String)> = tok
            .vocab
            .iter()
            .map(|e| (e.id, String::from_utf8_lossy(&e.str_bytes).to_string()))
            .collect();
        // Include added tokens (e.g. <think>, </think>, <|im_start|>)
        for added in &tok.added {
            entries.push((added.id, String::from_utf8_lossy(&added.str_bytes).to_string()));
        }
        entries.sort_by_key(|(id, _)| *id);
        let tokens: Vec<String> = entries.into_iter().map(|(_, s)| s).collect();
        Vocabulary { tokens }
    }

    fn decode(&self, token_id: usize) -> &str {
        self.tokens.get(token_id).map(|s| s.as_str()).unwrap_or("<unk>")
    }
}

// ─── Full layer forward ───────────────────────────────────────────────────

fn forward_layer(
    wf: &WeightFile, layer_idx: usize,
    hidden: &mut [f32], layer_states: &mut [LayerState], pos: usize,
    packed_fd: RawFd, ctx: Option<&MetalContext>, gpu_wf: Option<&GpuWeightCtx>,
    config: &ModelConfig,
) -> Result<(), MoEError> {
    let is_full_attn = (layer_idx + 1) % FULL_ATTN_INTERVAL == 0;

    if is_full_attn {
        if let LayerState::FullAttention(ref mut kv) = layer_states[layer_idx] {
            full_attention_forward(
                wf, layer_idx, hidden, kv, pos,
                config.hidden_dim,
                config.num_attn_heads, config.num_kv_heads,
                config.head_dim, config.rotary_dim,
                config.rope_theta, gpu_wf, ctx,
            );
        }
    } else {
        if let LayerState::LinearAttention(ref mut state) = layer_states[layer_idx] {
            gpu_forward::linear_attention_forward(
                wf, layer_idx, hidden, state,
                config.hidden_dim,
                config.linear_num_k_heads,
                config.linear_num_v_heads,
                config.linear_total_key,
                config.linear_total_value,
                config.linear_conv_dim,
                gpu_wf, ctx,
            );
        }
    }

    gpu_forward::moe_layer_forward(wf, layer_idx, hidden, packed_fd, ctx, gpu_wf, config)
}

// ─── Final norm ───────────────────────────────────────────────────────────

fn apply_final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    if let Some(fnw) = wf.get_tensor_u16("model.norm.weight") {
        let fnw_f32: Vec<f32> = fnw.iter().map(|&v| bf16_to_f32(v)).collect();
        let mut normed = vec![0.0f32; hidden_dim];
        cpu_rms_norm(hidden, &fnw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
        hidden.copy_from_slice(&normed);
    }
}

// ─── Model instance (loaded/cached per model name) ────────────────────────

struct ModelInstance {
    ctx: MetalContext,
    gpu_wf: GpuWeightCtx,
    wf: WeightFile,
    config: ModelConfig,
    tokenizer: BpeTokenizer,
    vocab: Vocabulary,
    layer_fds: Vec<RawFd>,
}

impl ModelInstance {
    fn load(model_dir: &Path) -> Result<Self, MoEError> {
        let config = load_model_config(model_dir)
            .map_err(|e| MoEError::Config(format!("config: {}", e)))?;

        let bin_path = model_dir.join("model_weights.bin");
        let json_path = model_dir.join("model_weights.json");
        let wf = WeightFile::open(&bin_path, &json_path)?;

        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::load(&tok_path)
            .map_err(|e| MoEError::Config(format!("tokenizer: {}", e)))?;
        let vocab = Vocabulary::from_tokenizer(&tokenizer);

        let ctx = MetalContext::init()?;
        let gpu_wf = GpuWeightCtx::new(&ctx.device, &wf);

        let packed_dir = model_dir.join("packed_experts");
        let mut layer_fds: Vec<RawFd> = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let path = packed_dir.join(format!("layer_{:02}.bin", layer));
            let file = std::fs::File::open(&path).map_err(|e| {
                MoEError::Io(std::io::Error::new(e.kind(),
                    format!("Cannot open layer {} expert file: {}", layer, e)))
            })?;
            layer_fds.push(file.into_raw_fd());
        }

        if layer_fds.is_empty() {
            return Err(MoEError::Config("No packed expert layer files found".into()));
        }

        eprintln!("[model] Loaded {} ({}, layers={}, dim={})",
            model_dir.file_name().unwrap_or_default().to_string_lossy(),
            model_dir.display(), config.num_layers, config.hidden_dim);

        Ok(ModelInstance { ctx, gpu_wf, wf, config, tokenizer, vocab, layer_fds })
    }

    fn new_layer_states(&self) -> Vec<LayerState> {
        let max_seq = 4096;
        (0..self.config.num_layers)
            .map(|layer| {
                if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                    LayerState::FullAttention(KVCache::new(
                        max_seq, self.config.head_dim, self.config.num_kv_heads))
                } else {
                    LayerState::LinearAttention(LinearAttnState::new(
                        self.config.linear_num_v_heads,
                        self.config.linear_total_key / self.config.linear_num_k_heads,
                        self.config.linear_total_value / self.config.linear_num_v_heads,
                        self.config.linear_conv_dim,
                    ))
                }
            })
            .collect()
    }
}

// ─── Server ───────────────────────────────────────────────────────────────

/// Run the HTTP inference server.
/// `data_dir` is the base directory — models are loaded from `data_dir/<model_name>/`.
pub fn run_server(port: u16, data_dir: &Path) -> Result<(), MoEError> {
    if !data_dir.exists() {
        return Err(MoEError::Config(format!("{} not found", data_dir.display())));
    }

    let model_cache: Mutex<HashMap<String, Arc<ModelInstance>>> = Mutex::new(HashMap::new());
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port))
        .map_err(|e| MoEError::Io(e))?;

    eprintln!("[serve] Listening on http://0.0.0.0:{}", port);
    eprintln!("[serve] Data dir: {}", data_dir.display());
    eprintln!("[serve] Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health");

    let req_counter = AtomicU64::new(0);

    for incoming in listener.incoming() {
        let mut stream = match incoming {
            Ok(s) => s,
            Err(e) => { eprintln!("[serve] accept error: {}", e); continue; }
        };

        let request_id = req_counter.fetch_add(1, Ordering::Relaxed);
        let rid = format!("req-{}", request_id);

        let req_bytes = match read_http_request(&mut stream) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let req_str = String::from_utf8_lossy(&req_bytes);
        let first_line = req_str.lines().next().unwrap_or("");
        let mut parts = first_line.split_whitespace();
        let method = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("");

        match (method, path) {
            ("OPTIONS", _) => {
                http_write_str(&stream, CORS_RESPONSE);
            }
            ("GET", "/health") => {
                let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}\n";
                http_write_str(&stream, resp);
            }
            ("GET", "/v1/models") => {
                // List available models in data_dir
                let mut models = Vec::new();
                if let Ok(entries) = std::fs::read_dir(data_dir) {
                    for entry in entries.flatten() {
                        if entry.path().join("config.json").exists() {
                            if let Some(name) = entry.file_name().to_str() {
                                models.push(format!("{{\"id\":\"{}\",\"object\":\"model\"}}", name));
                            }
                        }
                    }
                }
                let json_data = format!("{{\"object\":\"list\",\"data\":[{}]}}", models.join(","));
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}\n", json_data);
                http_write_str(&stream, &resp);
            }
            ("POST", "/v1/chat/completions") => {
                handle_chat_completion(
                    &mut stream, &rid, &req_str,
                    data_dir, &model_cache,
                );
            }
            _ => {
                let resp = "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{\"error\":\"not found\"}\n";
                http_write_str(&stream, resp);
            }
        }
    }

    Ok(())
}

// ─── Chat completion handler ──────────────────────────────────────────────

fn handle_chat_completion(
    stream: &mut TcpStream,
    request_id: &str,
    req_str: &str,
    data_dir: &Path,
    model_cache: &Mutex<HashMap<String, Arc<ModelInstance>>>,
) {
    let body_start = req_str.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    let body = &req_str[body_start..];

    let parsed: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => {
            http_write_str(stream, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"invalid json\"}\n");
            return;
        }
    };

    // Extract model name
    let model_name = parsed["model"].as_str().unwrap_or("");
    if model_name.is_empty() {
        http_write_str(stream, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"missing model\"}\n");
        return;
    }

    // Load or get cached model (Arc so we can clone and release the lock)
    let model_dir = data_dir.join(model_name);
    let model: Arc<ModelInstance> = {
        let cache = model_cache.lock().unwrap();
        if let Some(m) = cache.get(model_name) {
            Arc::clone(m)
        } else {
            drop(cache); // release lock during load
            match ModelInstance::load(&model_dir) {
                Ok(instance) => {
                    let arc = Arc::new(instance);
                    model_cache.lock().unwrap().insert(model_name.to_string(), Arc::clone(&arc));
                    arc
                }
                Err(e) => {
                    let err = format!(
                        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"error\":\"failed to load model {}: {}\"}}\n",
                        model_name, e
                    );
                    http_write_str(stream, &err);
                    return;
                }
            }
        }
    };

    let config = &model.config;
    let wf = &model.wf;
    let tokenizer = &model.tokenizer;
    let vocab = &model.vocab;
    let layer_fds = &model.layer_fds;
    let ctx = Some(&model.ctx);
    let gpu_wf = Some(&model.gpu_wf);
    let hidden_dim = config.hidden_dim;
    let num_layers = config.num_layers;

    // Tokenize full message history
    let max_tokens = parsed["max_tokens"].as_u64().unwrap_or(1024) as usize;

    eprintln!("[serve] {} model={} max_tokens={}",
        request_id, model_name, max_tokens);

    let prompt_ids = match tokenize_chat_messages(tokenizer, &parsed["messages"]) {
        Ok(ids) => ids,
        Err(e) => {
            let err = format!("HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"error\":\"{}\"}}\n", e);
            http_write_str(stream, &err);
            return;
        }
    };

    if prompt_ids.is_empty() {
        http_write_str(stream, "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"empty prompt\"}\n");
        return;
    }

    // Per-request state
    let mut layer_states = model.new_layer_states();
    let mut hidden = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        hidden[i] = (i as f32 * 0.1f32 + 0.3f32).sin() * 0.1f32;
    }

    // Send SSE headers
    http_write_str(stream, SSE_HEADERS);

    let t_start = Instant::now();
    let mut pos: usize = 0;

    // Pre-embed all tokens
    let mut embed_batch = vec![0.0f32; prompt_ids.len() * hidden_dim];
    for (i, &id) in prompt_ids.iter().enumerate() {
        embed_lookup(wf, id, &mut embed_batch[i * hidden_dim..(i + 1) * hidden_dim], hidden_dim);
    }
    {
        let e0 = &embed_batch[..hidden_dim.min(8)];
        let stats = stats_f32(&embed_batch[..hidden_dim]);
        eprintln!("[dbg] embed[0] first={:?} {:.2?}", e0, stats);
    }

    // Prefill intermediate tokens
    let n_prefill = prompt_ids.len().saturating_sub(1);
    for i in 0..n_prefill {
        hidden.copy_from_slice(&embed_batch[i * hidden_dim..(i + 1) * hidden_dim]);
        for layer in 0..num_layers {
            let _ = forward_layer(wf, layer, &mut hidden, &mut layer_states, pos,
                layer_fds[layer], ctx, gpu_wf, config);
        }
        pos += 1;
    }

    // Last prefill token
    if !prompt_ids.is_empty() {
        let last_i = prompt_ids.len() - 1;
        hidden.copy_from_slice(&embed_batch[last_i * hidden_dim..(last_i + 1) * hidden_dim]);
        for layer in 0..num_layers {
            let _ = forward_layer(wf, layer, &mut hidden, &mut layer_states, pos,
                layer_fds[layer], ctx, gpu_wf, config);
            if layer == 0 || (layer + 1) % 4 == 0 {
                let stats = stats_f32(&hidden[..hidden_dim.min(128)]);
                eprintln!("[dbg] after L{} hidden first8={:?} {:.2?}", layer, &hidden[..8], stats);
            }
        }
        pos += 1;
    }

    apply_final_norm(wf, &mut hidden, hidden_dim);
    {
        let stats = stats_f32(&hidden);
        eprintln!("[dbg] after final_norm hidden first8={:?} {:.2?}", &hidden[..8], stats);
    }

    let mut logits = vec![0.0f32; config.vocab_size];
    lm_head_forward(wf, &hidden, &mut logits, gpu_wf, ctx);
    {
        let stats = stats_f32(&logits);
        eprintln!("[dbg] logits first8={:?} {:.2?}", &logits[..8], stats);
    }
    let mut next_token = cpu_argmax(&logits);
    eprintln!("[dbg] next_token={} max_logit={:.4}", next_token, logits[next_token]);

    let mut gen_count = 0usize;

    for _gen in 0..max_tokens {
        if next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2 {
            embed_lookup(wf, next_token, &mut hidden, hidden_dim);
            for layer in 0..num_layers {
                let _ = forward_layer(wf, layer, &mut hidden, &mut layer_states, pos,
                    layer_fds[layer], ctx, gpu_wf, config);
            }
            break;
        }

        let tok_str = vocab.decode(next_token);
        if !sse_send_delta(stream, request_id, tok_str) {
            eprintln!("[serve] {} client disconnected", request_id);
            break;
        }
        gen_count += 1;

        embed_lookup(wf, next_token, &mut hidden, hidden_dim);
        for layer in 0..num_layers {
            let _ = forward_layer(wf, layer, &mut hidden, &mut layer_states, pos,
                layer_fds[layer], ctx, gpu_wf, config);
        }
        pos += 1;

        apply_final_norm(wf, &mut hidden, hidden_dim);
        logits.fill(0.0);
        lm_head_forward(wf, &hidden, &mut logits, gpu_wf, ctx);
        next_token = cpu_argmax(&logits);
    }

    sse_send_done(stream, request_id);

    let elapsed = t_start.elapsed().as_secs_f64() * 1000.0;
    let tok_s = if gen_count > 0 && elapsed > 0.0 {
        gen_count as f64 * 1000.0 / elapsed
    } else { 0.0 };
    eprintln!("[serve] {} generated={} tokens in {:.0}ms ({:.1} tok/s)",
        request_id, gen_count, elapsed, tok_s);
}
