use::burn::{
    module,
    prelude::Backend,
};
use tokenizers::{
    models::bpe::{BPE, BpeBuilder, Vocab, Merges},
    normalizers::NFC,
    tokenizer::{Tokenizer, Decoder}
};
use serde_json::Value;
use std::fs::{File, read_to_string};
use ahash::AHashMap;

const BOS_TOKEN: &str = "<BOS_TOKEN>";
const EOS_TOKEN: &str = "<|END_OF_TURN_TOKEN|>";
const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";


// TOKENIZER =============================
#[derive(Debug)]
pub struct TokenizerConfig {
    vocab_and_merges: Option<(String, String)>,
    tokenizer_json: Option<String>,
    use_default_system_prompt: bool,
    add_bos_token: bool,
    add_eos_token: bool,
    add_prefix_space: bool
}
 
impl Default for TokenizerConfig{
    fn default() -> Self {
        Self {
            vocab_and_merges: None,
            tokenizer_json: Some(String::from("./tokenizers.json")),
            use_default_system_prompt: true, add_bos_token: true, add_eos_token: false,
            add_prefix_space: false
        }   
    }
}

impl TokenizerConfig {
    pub fn change_tokenizer_file(&mut self, new_val: String){
        self.tokenizer_json = Some(new_val);
    }
}

//Dont need an associated function for this
pub fn tokenizer_builder(config: TokenizerConfig) -> Result<Tokenizer, Box<dyn std::error::Error>> {

    let v_and_m: (String, String) = match config.tokenizer_json {
        // get the vocab and merges extracted out of the tokenizers file
        Some(json_path) => {
            let file = File::open(json_path)?;
            let json_value: Value = serde_json::from_reader(file)?;

            let v: String = json_value
                .pointer("/model/vocab") // Option<&Value>
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            let m: String = json_value 
                .pointer("/model/merges") // Option<&Value>
                .and_then(|m| m.as_str().map(String::from))
                .unwrap_or_default();

            (v, m)
        }
        // read directly from files and load in as a String as well
        None => {
            match config.vocab_and_merges {
                Some((v_path,m_path)) => { (read_to_string(v_path)?, read_to_string(m_path)?) }
                None => { panic!("No file paths for tokenizers OR vocab+merges given to TokenizerConfig") }
            }
        }
    };

    // these two types are aliases in the tokenizers library, construct them from their original types
    let vocab: Vocab = AHashMap::new();
    let merges: Merges = vec![(String::from(""), String::from(""))];

    let bpe: BPE = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_normalizer(Some(NFC::default()));

    Ok(tokenizer)
}

fn from_tokenizer_json() -> () {
    
}
