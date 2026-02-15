pub(crate) fn configure_rayon_threads(num_threads: usize, debug_mode: bool) {
    if num_threads == 0 {
        return;
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(()) => {
            if debug_mode {
                eprintln!("Parallel: configured rayon worker threads={num_threads}");
            }
        }
        Err(e) => {
            if debug_mode {
                eprintln!(
                    "Parallel: keeping existing rayon global thread pool (requested {num_threads}, reason: {e})"
                );
            }
        }
    }
}
