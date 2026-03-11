mod app;
mod cli;
mod engine;
mod tools;
mod vendors;

fn main() {
    if let Err(e) = app::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
