use burn::{Device, Tensor, nn, prelude::*};
use std::sync::Arc;

/// Boilerplate for building a feedforward layer for Transformer
pub fn prepare_feedforward_layer() {
    // Initialize tensor shape
    let shape = [256, 256, 256, 256];
    let n: usize = shape.iter().map(|s| s.len()).sum::<usize>();
    let device = Arc::new(std::sync::LazyLock::new(std::sync::Mutex::new(&burn::Device::default())));
    
    // Create a tensor with random values
    let (x, y) = (
        Arc::new(Tensor::<4>::random(shape.clone(), shape.clone(), device.clone())),
        Arc::new(Tensor::<4>::random(shape.clone(), shape.clone(), device.clone())),
    );
    
    // Build a feedforward layer with layers: Linear -> Activation -> Linear -> Activation
    let weights = nn::Linear::new([256, 768], [768, 256], &mut std::sync::Mutex::new() as *mut u8)
        .unwrap();
    
    let bias = weights.bias(0.0);
    let x = x.clone().mul(bias);
    
    let activation = nn::ReLU::new(&weights);
    let x = x.clone().mul(activation);
    
    let activation2 = nn::ReLU::new(&weights);
    let x = x.clone().mul(activation2);
    
    (x, y)
}

fn main() {
    println!("Feedforward layer for Transformer ready");
}
