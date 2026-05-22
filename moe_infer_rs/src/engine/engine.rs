use crate::cache::Cache;
use crate::math::SignalCheckFn;

pub trait Engine {
    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String>;
}
