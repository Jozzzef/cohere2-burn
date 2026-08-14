use burn::config::Config;

#[derive(Config, Debug)]
pub struct Cohere2Config {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
}

