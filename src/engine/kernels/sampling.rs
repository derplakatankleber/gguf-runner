use crate::engine::types::XorShiftRng;
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

pub(crate) struct TopKSampler {
    candidates: Vec<Candidate>,
    probs: Vec<f32>,
}

impl TopKSampler {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            probs: Vec::new(),
        }
    }

    fn find_min_pos(cands: &[Candidate]) -> usize {
        let mut min_pos = 0usize;
        let mut min_score = cands[0].score;
        for (i, c) in cands.iter().enumerate().skip(1) {
            if c.score < min_score {
                min_score = c.score;
                min_pos = i;
            }
        }
        min_pos
    }

    pub(crate) fn sample_top_k_top_p(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng: &mut XorShiftRng,
    ) -> usize {
        let k = top_k.min(logits.len()).max(1);
        self.candidates.clear();
        if self.candidates.capacity() < k {
            self.candidates.reserve(k - self.candidates.capacity());
        }

        let mut min_pos = 0usize;
        for (idx, &logit) in logits.iter().enumerate() {
            let score = logit / temperature;
            if self.candidates.len() < k {
                self.candidates.push(Candidate { idx, score });
                min_pos = Self::find_min_pos(&self.candidates);
            } else if score > self.candidates[min_pos].score {
                self.candidates[min_pos] = Candidate { idx, score };
                min_pos = Self::find_min_pos(&self.candidates);
            }
        }

        self.candidates
            .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

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
        if top_p < 1.0 {
            let mut cumulative = 0.0f32;
            keep = 0;
            for &p in &self.probs {
                cumulative += p / prob_sum;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
        }

        let kept_sum: f32 = self.probs[..keep].iter().copied().sum();
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
