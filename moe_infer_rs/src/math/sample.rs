use rand::Rng;

use crate::math::softmax;

pub fn sample(logits: &mut [f32], temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> usize {
    let n = logits.len();
    if (temperature - 1.0).abs() > 1e-7 {
        let inv = 1.0 / temperature.max(1e-8);
        for v in logits.iter_mut() { *v *= inv; }
    }
    if temperature < 0.01 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }
    softmax(logits);

    if top_k > 0 && top_k < n {
        let mut v: Vec<f32> = logits.to_vec();
        v.select_nth_unstable_by(top_k, |a, b| b.partial_cmp(a).unwrap());
        let t = v[top_k - 1];
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }
    if top_p < 1.0 {
        let mut s: Vec<f32> = logits.iter().copied().filter(|&x| x > 0.0).collect();
        s.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let total: f32 = s.iter().sum();
        let mut cum = 0.0;
        let mut cut = 0.0;
        for v in s {
            cum += v;
            if cum / total >= top_p { cut = v; break; }
        }
        for x in logits.iter_mut() { if *x < cut { *x = 0.0; } }
    }
    if min_p > 0.0 {
        let max_p = logits.iter().fold(0.0f32, |a, &b| a.max(b));
        let t = max_p * min_p;
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }

    let sum: f32 = logits.iter().sum();
    if sum <= 0.0 { return 0; }
    let inv = 1.0 / sum;
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0;
    for (i, &v) in logits.iter().enumerate() {
        cum += v * inv;
        if r <= cum { return i; }
    }
    n - 1
}
