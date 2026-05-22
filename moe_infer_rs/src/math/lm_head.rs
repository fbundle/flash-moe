use crate::constants::GROUP_SIZE;
use crate::metal_context::{metal_buf_shared, WeightBuffer, MetalContext};
use crate::model::weights::WeightFile;
use crate::math::dequant_matvec_4bit;

/// GPU lm_head matvec.
pub fn gpu_lm_head(
    wf: &WeightFile, hidden: &[f32], logits: &mut [f32],
    gpu_wf: &WeightBuffer, ctx: &MetalContext,
) {
    let x_buf = metal_buf_shared(&ctx.device, hidden.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), x_buf.contents() as *mut f32, hidden.len());
    }
    let out_buf = metal_buf_shared(&ctx.device, logits.len() * 4);
    let cm = ctx.queue.new_command_buffer();
    let enc = cm.new_compute_command_encoder();
    gpu_wf.encode_matvec_into(wf, ctx, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
    enc.end_encoding();
    cm.commit();
    cm.wait_until_completed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len());
    }
}

/// CPU-only lm_head: dequant matvec for vocab projection.
pub fn lm_head(wf: &WeightFile, hidden: &[f32], logits: &mut [f32]) {
    let prefix = "lm_head";
    let hd = hidden.len();
    if let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32(&format!("{}.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.biases", prefix)),
    ) {
        dequant_matvec_4bit(w, s, b, hidden, logits, logits.len(), hd, GROUP_SIZE);
    }
}
