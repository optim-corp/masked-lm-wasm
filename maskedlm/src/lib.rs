extern crate console_error_panic_hook;
use std::panic;
use std::collections::HashMap;
use std::io::BufReader;
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput, Encoding, AddedToken};
use tokenizers::*;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::*;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub fn predict_masked_words(word: &str) -> String {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    run_predict(word).unwrap()
}

fn run_predict(word: &str) -> Result<String> {
    let tokenizer = create_tokenizer()?;
    let encoding = tokenizer.encode(EncodeInput::Single(word.into()), true)?;
    let mask_positions = get_mask_position(&encoding);
    let output = inference(
        create_input_tensor(&encoding)?,
        encoding.get_ids().len()
        )?;
    decode(&output, &tokenizer, mask_positions, word)
}

fn create_tokenizer() -> Result<Tokenizer> {

    // Normalizer
    let normalizer = normalizers::bert::BertNormalizer::new(true, false, false, false);

    // PreTokenizer
    let pre_tokenizer = pre_tokenizers::bert::BertPreTokenizer;

    // Model
    let vocab_str = include_str!("../vocab.txt");
    let mut vocab = HashMap::new();
    for (index, line) in vocab_str.lines().enumerate() {
        vocab.insert(line.trim_end().to_owned(), index as u32);
    }
    let wordpiece_builder = models::wordpiece::WordPiece::builder();
    let wordpiece = wordpiece_builder
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build().unwrap();

    // Post processor
    let post_processor = processors::bert::BertProcessing::new(("[SEP]".into(), 102), ("[CLS]".into(), 101));

    // build Tokenizer
    let mut tokenizer = Tokenizer::new(Box::new(wordpiece));
    tokenizer.with_normalizer(Box::new(normalizer));
    tokenizer.with_pre_tokenizer(Box::new(pre_tokenizer));
    tokenizer.with_post_processor(Box::new(post_processor));

    // add [MASK] token
    let mask_token = AddedToken::from("[MASK]".into()).single_word(true);
    tokenizer.add_special_tokens(&[mask_token]);

    // tokenize
    Ok(tokenizer)
}


fn get_mask_position(encoding: &Encoding) -> Vec<usize> {
    let mut mask_positions = Vec::new();
    for (i, token) in  encoding.get_tokens().into_iter().enumerate() {
        if token == "[MASK]" {
            mask_positions.push(i);
        }
    }
    mask_positions
}


fn create_input_tensor(encoding: &Encoding) -> Result<TVec<Tensor>> {

    // convert to Tensor
    fn element2tensor(element: &[u32]) -> Result<Tensor> {
        let e_i64: Vec<i64> = element.into_iter().map(|&e| e as i64).collect();
        Ok(tract_ndarray::Array::from_shape_vec((1, e_i64.len()), e_i64)?.into())
    }

    // prepare input tensor 
    let ids: Tensor = element2tensor(encoding.get_ids())?;
    let attention_mask: Tensor = element2tensor(encoding.get_attention_mask())?;
    let type_ids: Tensor = element2tensor(encoding.get_type_ids())?;
    let input_tensor = tvec![ids, attention_mask, type_ids];

    Ok(input_tensor)
}


fn inference(input_tensor: TVec<Tensor>, seq_length: usize) -> Result<Array<f32, Dim<[usize; 2]>>> {

    // load model
    let onnx_model = include_bytes!("../bert-masked.onnx");
    let model = tract_onnx::onnx()
        .model_for_read(&mut BufReader::new(&onnx_model[..]))?
        .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_length)))?
        .with_input_fact(1, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_length)))?
        .with_input_fact(2, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, seq_length)))?
        .into_optimized()?
        .into_runnable()?;

    // inference
    let output = model.run(input_tensor)?[0]
        .to_array_view::<f32>()?
        .slice(s![0, .., ..])
        .into_owned();

    Ok(output)
}


fn argmax<T: PartialOrd>(v: &[T]) -> usize {
    if v.len() == 1 {
        0
    } else {
        let mut maxval = &v[0];
        let mut max_idx: usize = 0;
        for (i, x) in v.iter().enumerate().skip(1) {
            if x > maxval {
                maxval = x;
                max_idx = i;
            }
        }
        max_idx
    }
}


fn decode(output: &Array<f32, Dim<[usize; 2]>>, tokenizer: &Tokenizer, mask_positions: Vec<usize>, word: &str) -> Result<String> {

    let mut decoded: String = word.into();
    for i in mask_positions {
        // predict masked ids by argmax
        let prediction = output.slice(s![i, ..]);
        let prediction = prediction.as_slice().ok_or("Output is invalid")?;
        let max_id: u32 = argmax(prediction) as u32;
        // decode id to word
        let word = tokenizer.decode(vec![max_id], false)?;
        // replace ids
        decoded = decoded.replacen("[MASK]", &word[..], 1);
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_predict() {
        let predicted = run_predict("Rust is the best programming [MASK].").unwrap();
        assert_eq!("Rust is the best programming language.".to_string(), predicted);
    }
}
