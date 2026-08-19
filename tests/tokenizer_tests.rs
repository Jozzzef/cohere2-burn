
#[test]
fn debug_file_load(){
    use cohere2_burn::tokenization::{TokenizerConfig, tokenizer_builder};

    let mut config: TokenizerConfig = TokenizerConfig::default();
    config.change_tokenizer_file(String::from("./assets/tokenizers.json"));
    println!("{:?}", config);
}
