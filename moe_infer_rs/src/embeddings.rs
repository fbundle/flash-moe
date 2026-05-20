// Embedding lookup and LM head — mirrors embeddings.h

use crate::kernels::{bf16_to_f32, cpu_dequant_matvec};
use crate::types::*;
use crate::weights::OwnedTensorHashTable;

/// 4-bit quantized embedding lookup: out[hidden_dim] = embed_table[token_id]
pub fn embed_lookup(
    wf: &WeightFile,
    ht: &OwnedTensorHashTable,
    token_id: i32,
    hidden_dim: i32,
    out: &mut [f32],
) {
    let w_info = match ht.find("model.embed_tokens.weight") {
        Some(t) => t,
        None => {
            eprintln!("ERROR: embedding tensors not found");
            out.fill(0.0);
            return;
        }
    };
    let s_info = ht.find("model.embed_tokens.scales").unwrap();
    let b_info = ht.find("model.embed_tokens.biases").unwrap();

    let packed_cols = w_info.shape[1] as usize;
    let num_groups = s_info.shape[1] as usize;
    let group_size = hidden_dim as usize / num_groups;
    let packed_per_group = group_size / 8;

    let w_ptr = unsafe { wf.data.add(w_info.offset as usize) as *const u32 };
    let s_ptr = unsafe { wf.data.add(s_info.offset as usize) as *const u16 };
    let b_ptr = unsafe { wf.data.add(b_info.offset as usize) as *const u16 };

    let w_row = unsafe { std::slice::from_raw_parts(w_ptr.add(token_id as usize * packed_cols), packed_cols) };
    let s_row = unsafe { std::slice::from_raw_parts(s_ptr.add(token_id as usize * num_groups), num_groups) };
    let b_row = unsafe { std::slice::from_raw_parts(b_ptr.add(token_id as usize * num_groups), num_groups) };

    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        for p in 0..packed_per_group {
            let packed = w_row[g * packed_per_group + p];
            let base = g * group_size + p * 8;
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = nibble as f32 * scale + bias;
            }
        }
    }
}

/// LM head: logits[vocab_size] = lm_head_weight * hidden[hidden_dim]
pub fn lm_head_forward(
    wf: &WeightFile,
    ht: &OwnedTensorHashTable,
    hidden: &[f32],
    logits: &mut [f32],
    vocab_size: i32,
    hidden_dim: i32,
    group_size: i32,
    metal_ctx: Option<&crate::metal::MetalCtx>,
) {
    let w_info = ht.find("lm_head.weight").expect("lm_head.weight not found");
    let s_info = ht.find("lm_head.scales").expect("lm_head.scales not found");
    let b_info = ht.find("lm_head.biases").expect("lm_head.biases not found");

    let w_ptr = unsafe { wf.data.add(w_info.offset as usize) as *const u32 };
    let s_ptr = unsafe { wf.data.add(s_info.offset as usize) as *const u16 };
    let b_ptr = unsafe { wf.data.add(b_info.offset as usize) as *const u16 };

    if let Some(ctx) = metal_ctx {
        unsafe {
            crate::gpu_ops::fast_dequant_matvec(
                ctx,
                w_ptr, s_ptr, b_ptr,
                hidden, logits,
                vocab_size as usize, hidden_dim as usize, group_size as usize,
            );
        }
    } else {
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, (vocab_size as usize * hidden_dim as usize) / 8) };
        let s_slice = unsafe {
            std::slice::from_raw_parts(
                s_ptr,
                vocab_size as usize * (hidden_dim as usize / group_size as usize),
            )
        };
        let b_slice = unsafe {
            std::slice::from_raw_parts(
                b_ptr,
                vocab_size as usize * (hidden_dim as usize / group_size as usize),
            )
        };

        cpu_dequant_matvec(
            w_slice, s_slice, b_slice,
            hidden, logits,
            vocab_size as usize, hidden_dim as usize, group_size as usize,
        );
    }
}
