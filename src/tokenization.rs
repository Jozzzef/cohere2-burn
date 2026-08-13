use::burn::{
    module,
    tensor::Backend,
};
use tokenizers::{
    tokenizer::Tokenizer,
    tokenizers::{
        decoders::Decoder,
        models::BPEBuilder,
        normalizers::NFC,
        pre_tokenizers::{Digits, ByteLevel, PreTokenizerWrapper}
    },
};

pub struct TokenizerConfig {
   vocab: ,
   merges: ,
}

pub fn tokenizer_builder() -> Tokenizer {

}
