mod agent;
mod generation;
mod tools;

use crate::cli::CliOptions;
use crate::engine::profiling::{print_profile_report, profiling_reset, set_profiling_enabled};
use crate::engine::switches::{
    init_runtime_config, par_attn_min_heads, par_matmul_chunk_rows, par_matmul_min_rows,
    par_qwen3next_min_heads, RuntimeSwitchConfig,
};
use std::time::Instant;

pub(crate) fn run() -> Result<(), String> {
    let cli = CliOptions::parse()?;
    let runtime_switch_config = RuntimeSwitchConfig {
        par_matmul_min_rows: cli.par_matmul_min_rows,
        par_matmul_chunk_rows: cli.par_matmul_chunk_rows,
        par_attn_min_heads: cli.par_attn_min_heads,
        par_qwen3next_min_heads: cli.par_qwen3next_min_heads,
        #[cfg(target_arch = "aarch64")]
        aarch64_dotprod_q8: cli.aarch64_dotprod_q8,
        #[cfg(target_arch = "aarch64")]
        aarch64_qk_mr4: cli.aarch64_qk_mr4,
        #[cfg(target_arch = "x86_64")]
        x86_avx2: cli.x86_avx2,
        #[cfg(target_arch = "x86_64")]
        x86_f16c: cli.x86_f16c,
        #[cfg(target_arch = "x86_64")]
        x86_qk_mr4: cli.x86_qk_mr4,
        #[cfg(target_arch = "x86_64")]
        x86_avxvnni: cli.x86_avxvnni,
        #[cfg(target_arch = "x86_64")]
        x86_avx512vnni_q8: cli.x86_avx512vnni_q8,
        layer_debug: cli.layer_debug,
        layer_debug_pos: cli.layer_debug_pos,
    };
    init_runtime_config(&runtime_switch_config);
    let run_started = Instant::now();

    set_profiling_enabled(cli.profiling);
    if cli.profiling {
        profiling_reset();
    }

    if cli.debug {
        eprintln!(
            "Parallel thresholds: matmul_min_rows={}, matmul_chunk_rows={}, attn_min_heads={}, qwen3next_min_heads={}",
            par_matmul_min_rows(),
            par_matmul_chunk_rows(),
            par_attn_min_heads(),
            par_qwen3next_min_heads()
        );
    }

    let mut runtime = generation::ModelRuntime::load(&cli)?;
    if cli.agent {
        agent::run_agent_loop(&mut runtime, &cli)?;
    } else {
        let _ = runtime.generate_text(&cli.prompt, &cli.system_prompt, true)?;
    }

    if cli.profiling {
        print_profile_report();
    }
    if cli.show_timings {
        eprintln!(
            "overall runtime: {:.3}s",
            run_started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}
