/// Generation loop: autoregressive token generation using any Engine impl.
use std::collections::HashSet;
use std::time::Instant;

use crate::engine::{Cache, Engine};
use crate::math::sample;

// ─── Telemetry ──────────────────────────────────────────────────────────────

pub struct Telemetry {
    pub prefill_ms: f64,
    pub total_ms: f64,
    pub tokens_generated: usize,
}

// ─── Sample params ──────────────────────────────────────────────────────────

pub struct SampleParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub eos: HashSet<usize>,
}

/// Run autoregressive generation: prefix forward + sampling loop.
///
/// Returns `(tokens, last_logits)` and updates `telemetry` with timing.
pub fn generate(
    engine: &mut dyn Engine,
    input_ids: &[i64],
    cache: &mut Cache,
    params: &SampleParams,
    check_signal: &mut dyn FnMut() -> bool,
    telemetry: &mut Telemetry,
) -> Result<(Vec<i64>, Vec<f32>), String> {
    let gen_t0 = Instant::now();
    let n = input_ids.len();

    // ── Prefix forward ─────────────────────────────────────────────────
    let logits = engine.forward(input_ids, cache, &mut || check_signal())?;
    let vs = if n > 0 { logits.len() / n } else { 0 };

    let mut logits_last = if n > 0 {
        logits[logits.len() - vs..].to_vec()
    } else {
        vec![0.0f32; vs]
    };

    // ── Sample first token ─────────────────────────────────────────────
    let mut next = if n > 0 {
        pick_token(&logits_last, params)
    } else {
        0
    };

    // ── Autoregressive loop ────────────────────────────────────────────
    let mut output = Vec::with_capacity(params.max_tokens);
    for _ in 0..params.max_tokens {
        if params.eos.contains(&next) {
            break;
        }
        output.push(next as i64);

        logits_last = engine.forward(&[next as i64], cache, &mut || check_signal())?;
        next = pick_token(&logits_last, params);
    }

    telemetry.total_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
    telemetry.tokens_generated = output.len();
    Ok((output, logits_last))
}

fn pick_token(logits: &[f32], params: &SampleParams) -> usize {
    if params.temperature < 0.01 {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        let mut copy = logits.to_vec();
        sample(&mut copy, params.temperature, params.top_k, params.top_p, params.min_p)
    }
}
