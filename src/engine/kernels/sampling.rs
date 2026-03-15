use crate::engine::types::XorShiftRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
pub(crate) fn argmax(v: &[f32]) -> usize {
    let mut max_i = 0usize;
    let mut max_p = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_p {
            max_p = val;
            max_i = i;
        }
    }
    max_i
}

pub(crate) fn sample(probabilities: &[f32], rng: &mut XorShiftRng) -> usize {
    let r = rng.random_f32();
    let mut cdf = 0.0f32;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probabilities.len().saturating_sub(1)
}

#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    idx: usize,
    score: f32,
}

#[derive(Clone, Copy)]
struct HeapCandidate {
    idx: usize,
    score: f32,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.score.total_cmp(&other.score) == Ordering::Equal
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse score ordering so BinaryHeap acts like a min-heap by score.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

pub(crate) struct TopKSampler {
    candidates: Vec<Candidate>,
    probs: Vec<f32>,
    heap: BinaryHeap<HeapCandidate>,
}

impl TopKSampler {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            probs: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    pub(crate) fn sample_top_k_top_p(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut XorShiftRng,
    ) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let k = top_k.min(logits.len()).max(1);
        self.heap.clear();
        if self.heap.capacity() < k {
            self.heap.reserve(k - self.heap.capacity());
        }
        let inv_temperature = 1.0 / temperature;
        for (idx, &logit) in logits.iter().enumerate() {
            let score = logit * inv_temperature;
            if self.heap.len() < k {
                self.heap.push(HeapCandidate { idx, score });
                continue;
            }
            if let Some(min_keep) = self.heap.peek()
                && score > min_keep.score
            {
                let _ = self.heap.pop();
                self.heap.push(HeapCandidate { idx, score });
            }
        }

        // With k=1 this is deterministic argmax-equivalent and avoids extra work.
        if k == 1 {
            return self.heap.peek().map(|c| c.idx).unwrap_or(0);
        }

        self.candidates.clear();
        if self.candidates.capacity() < self.heap.len() {
            self.candidates
                .reserve(self.heap.len() - self.candidates.capacity());
        }
        while let Some(c) = self.heap.pop() {
            self.candidates.push(Candidate {
                idx: c.idx,
                score: c.score,
            });
        }
        // Min-heap pop order is low->high; sampling expects high->low.
        self.candidates.reverse();

        let max_score = self.candidates[0].score;
        self.probs.clear();
        if self.probs.capacity() < self.candidates.len() {
            self.probs
                .reserve(self.candidates.len() - self.probs.capacity());
        }

        let mut prob_sum = 0.0f32;
        for c in &self.candidates {
            let p = (c.score - max_score).exp();
            self.probs.push(p);
            prob_sum += p;
        }

        let mut keep = self.candidates.len();
        let mut kept_sum = prob_sum;
        if top_p < 1.0 {
            let mut cumulative_raw = 0.0f32;
            keep = 0;
            for &p in &self.probs {
                cumulative_raw += p;
                keep += 1;
                if cumulative_raw >= top_p * prob_sum {
                    break;
                }
            }
            kept_sum = cumulative_raw;
        }
        keep = keep.max(1);

        let mut r = rng.random_f32() * kept_sum;
        for i in 0..keep {
            r -= self.probs[i];
            if r <= 0.0 {
                return self.candidates[i].idx;
            }
        }
        self.candidates[keep - 1].idx
    }
}

#[cfg(test)]
mod tests {
    use super::{TopKSampler, XorShiftRng};

    #[test]
    fn top_k_one_is_argmax_equivalent() {
        let logits = [0.1f32, 2.5, 1.3, -0.4];
        let mut sampler = TopKSampler::new();
        let mut rng = XorShiftRng::new(123);
        for _ in 0..8 {
            let tok = sampler.sample_top_k_top_p(&logits, 0.7, 1, 0.95, &mut rng);
            assert_eq!(tok, 1);
        }
    }

    #[test]
    fn top_k_limits_candidate_set() {
        let logits = [0.1f32, 0.2, 4.0, 5.0, -1.0];
        let mut sampler = TopKSampler::new();
        let mut rng = XorShiftRng::new(42);
        for _ in 0..64 {
            let tok = sampler.sample_top_k_top_p(&logits, 1.0, 2, 1.0, &mut rng);
            assert!(tok == 2 || tok == 3);
        }
    }

    #[test]
    fn top_p_can_force_single_token() {
        let logits = [10.0f32, 0.0, 0.0, 0.0];
        let mut sampler = TopKSampler::new();
        let mut rng = XorShiftRng::new(7);
        for _ in 0..16 {
            let tok = sampler.sample_top_k_top_p(&logits, 1.0, 4, 0.5, &mut rng);
            assert_eq!(tok, 0);
        }
    }
}
