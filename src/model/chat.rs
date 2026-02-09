use crate::{
    Config, Tokenizer, GEMMA3_BOS_TOKEN, GEMMA3_END_TURN, GEMMA3_START_TURN, LLAMA3_BOS_TOKEN,
    LLAMA3_END_HEADER, LLAMA3_EOT, LLAMA3_START_HEADER,
};

pub(crate) fn encode_chat_prompt(
    tokenizer: &mut Tokenizer,
    config: &Config,
    prompt: &str,
    system_prompt: &str,
) -> Vec<i32> {
    if config.is_gemma3 {
        encode_gemma3_chat(tokenizer, prompt, system_prompt)
    } else if config.is_qwen3moe || config.is_qwen3next {
        encode_qwen3_chat(tokenizer, prompt, system_prompt)
    } else if config.is_qwen2 {
        encode_qwen2_chat(tokenizer, prompt, system_prompt)
    } else {
        encode_llama3_chat(tokenizer, prompt, system_prompt)
    }
}

fn encode_gemma3_chat(tokenizer: &mut Tokenizer, prompt: &str, system_prompt: &str) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos_token = tokenizer
        .find_special_token("<bos>")
        .unwrap_or(GEMMA3_BOS_TOKEN);
    let start_turn = tokenizer
        .find_special_token("<start_of_turn>")
        .unwrap_or(GEMMA3_START_TURN);
    let end_turn = tokenizer
        .find_special_token("<end_of_turn>")
        .unwrap_or(GEMMA3_END_TURN);

    tokens.push(bos_token);

    let full_prompt = if !system_prompt.is_empty() {
        format!("{}\n\n{}", system_prompt, prompt)
    } else {
        prompt.to_string()
    };

    tokens.push(start_turn);
    tokenizer.bpe_encode(&format!("user\n{}", full_prompt), &mut temp);
    tokens.extend_from_slice(&temp);

    tokens.push(end_turn);
    tokenizer.bpe_encode("\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens.push(start_turn);
    tokenizer.bpe_encode("model\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}

fn encode_llama3_chat(tokenizer: &mut Tokenizer, prompt: &str, system_prompt: &str) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);

    let bos = tokenizer
        .find_special_token("<|begin_of_text|>")
        .unwrap_or(LLAMA3_BOS_TOKEN);
    let start_header = tokenizer
        .find_special_token("<|start_header_id|>")
        .unwrap_or(LLAMA3_START_HEADER);
    let end_header = tokenizer
        .find_special_token("<|end_header_id|>")
        .unwrap_or(LLAMA3_END_HEADER);
    let eot = tokenizer
        .find_special_token("<|eot_id|>")
        .unwrap_or(LLAMA3_EOT);

    tokens.push(bos);

    if !system_prompt.is_empty() {
        tokens.push(start_header);
        tokenizer.bpe_encode("system", &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end_header);
        tokenizer.bpe_encode(&format!("\n\n{}", system_prompt), &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(eot);
    }

    tokens.push(start_header);
    tokenizer.bpe_encode("user", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode(&format!("\n\n{}", prompt), &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(eot);

    tokens.push(start_header);
    tokenizer.bpe_encode("assistant", &mut temp);
    tokens.extend_from_slice(&temp);
    tokens.push(end_header);
    tokenizer.bpe_encode("\n\n", &mut temp);
    tokens.extend_from_slice(&temp);

    tokens
}

fn encode_qwen2_chat(tokenizer: &mut Tokenizer, prompt: &str, system_prompt: &str) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = if system_prompt.is_empty() {
        "You are a helpful assistant."
    } else {
        system_prompt
    };

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        tokens.push(start);
        tokenizer.bpe_encode("system\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(sys, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        return tokens;
    }

    // Fallback: encode ChatML markers as plain text if special tokens are not mapped.
    let rendered = format!(
        "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}

fn encode_qwen3_chat(tokenizer: &mut Tokenizer, prompt: &str, system_prompt: &str) -> Vec<i32> {
    let mut tokens: Vec<i32> = Vec::with_capacity(8192);
    let mut temp: Vec<i32> = Vec::with_capacity(8192);
    let sys = system_prompt.trim();

    let im_start = tokenizer.find_special_token("<|im_start|>");
    let im_end = tokenizer.find_special_token("<|im_end|>");

    if tokenizer.bos_token >= 0 {
        tokens.push(tokenizer.bos_token);
    }

    if let (Some(start), Some(end)) = (im_start, im_end) {
        if !sys.is_empty() {
            tokens.push(start);
            tokenizer.bpe_encode("system\n", &mut temp);
            tokens.extend_from_slice(&temp);
            tokenizer.bpe_encode(sys, &mut temp);
            tokens.extend_from_slice(&temp);
            tokens.push(end);
            tokenizer.bpe_encode("\n", &mut temp);
            tokens.extend_from_slice(&temp);
        }

        tokens.push(start);
        tokenizer.bpe_encode("user\n", &mut temp);
        tokens.extend_from_slice(&temp);
        tokenizer.bpe_encode(prompt, &mut temp);
        tokens.extend_from_slice(&temp);
        tokens.push(end);
        tokenizer.bpe_encode("\n", &mut temp);
        tokens.extend_from_slice(&temp);

        tokens.push(start);
        tokenizer.bpe_encode("assistant\n", &mut temp);
        tokens.extend_from_slice(&temp);
        return tokens;
    }

    let rendered = if sys.is_empty() {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        format!(
            "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    };
    tokenizer.bpe_encode(&rendered, &mut tokens);
    tokens
}
