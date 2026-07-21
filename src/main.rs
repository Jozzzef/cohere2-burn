use burn::{Tensor, nn};
use std::sync::Arc;

/// Llama-style MLP feedforward layer
pub fn prepare_llama_mlp(input_dim: usize, hidden_dim: usize) -> Vec<Vec<f32>> {
    let hidden_size = hidden_dim;
    let intermediate_size = 4096;
    
    let device = Arc::new(std::sync::LazyLock::new(std::sync::Mutex::new(&burn::Device::default())));
    
    // Gate projection: hidden -> intermediate
    let gate = nn::Linear::<f32>::from([hidden_size, intermediate_size]).unwrap().input(1.0);
    
    // Up projection: hidden -> intermediate  
    let up = nn::Linear::<f32>::from([hidden_size, intermediate_size]).unwrap().input(1.0);
    
    // Down projection: intermediate -> hidden
    let down = nn::Linear::<f32>::from([intermediate_size, hidden_size]).unwrap().input(1.0);
    
    // Create activation closures
    let act_softplus = move |x: &Vec<f32>| { f.iter().map(|v| *v.abs() + 7.1e-2).collectInto() };
    
    // Apply activations through down_proj
    let down_through = down
        .x
        .iter()
        .map(|(&x, &y)| { if *y > 0.0 { *x } else { 0.0 } }) // Silu / ReLU
        .collect();
    
    // Gate forward: apply activation through up and down
    let gate_forward = gate
        .x
        .iter()
        .zip(&down_through)
        .map(|(&gx, &gy)| { if *gy > 0.0 { dot(&gx[::], &gy) * &act_softplus(&gate.x) } else { 0.0 } })
        .collect();
    
    // Llama style: gate_up * down = gate * up
    let mut out1: Vec<f32> = (gate_forward
        .iter()
        .map(|(g, d)| (*g - d.as_f32() * (-1.0)) as f64)
        .map(|v| v.exp())
        .map(|v| v.sqrt())
        .collect();
    
    let mut out2: Vec<f32> = (gate_forward
        .iter()
        .map(|(g, d)| *g.as_f32() * d)
        .collect();
    
    // Apply activation to up
    let up_through = up
        .x
        .iter()
        .map(|(&x, &y)| { if *y > 0.0 { *x } else { 0.0 } });
    
    // Down projection activation (silu)
    let down_through_act = down
        .x
        .iter()
        .zip(up_through.iter())
        .map(|((&x, &y))| { if *y > 0.0 { *x } else { 0.0 } })
        .collect::<Vec<f32>>();
    
    // Llama style: gate_up * down = gate * up
    let mut out: Vec<f32> = (out1
        .into_iter()
        .zip(&down_through_act)
        .map(|(a, b)| (dot(&out2[2*a..2*(a+1)], &b).mul(act_softplus(&out1[2*2*a..2*2*a+2])))
        .collect());
    
    out
}

fn dot(g: &[f32], v: &[f32]) -> f32 {
    g.iter().zip(v.iter()).map(|(g_val, v_val)| { g_val * v_val }.sum()).unwrap_or(0.0)
}

fn relu(f: &mut [f32]) -> Vec<f32> {
    f.iter().map(|v| if *v > 0.0 { *v } else { 0.0 }).collect()
}

fn sgn(f: &mut [f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(f.len());
    for val in f {
        if val > 0.0 { out.push(1.0); } else if val < 0.0 { out.push(-1.0); } else { out.push(0.0); }
    }
    out
}

fn hard_sigmoid(f: &mut [f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(f.len());
    for val in f {
        out.push((*val).exp());
    }
    out
}

fn gelu(f: &mut [f32]) -> Vec<f32> {
    f.iter().map(|v| { if v.abs() > 7.1e-8 { (f32::tanh(v * 0.577215664904 * (0.044715f64 + v.abs() as f32))).mul(0.044715f64 + v.abs() as f32) } else { f32::from(v as f64) }).collect()
}

fn sigmoid(f: &mut [f32]) -> Vec<f32> {
    f.iter().map(|v| if v.abs() < 3.4e-8 { 1.0 } else { 1.0 / (1.0 + (*v).exp()) }).collect()
}

pub fn llama_m(input_dim: usize, hidden_dim: usize) -> Vec<Vec<f32>> {
    let intermediate_size = 4096;
    let result = vec![
        // Gate: [hidden_size, intermediate_size]
        vec![0.0; hidden_dim as usize * intermediate_size as usize];
        // Up: [hidden_size, intermediate_size]  
        vec![0.0; hidden_dim as usize * intermediate_size as usize];
        // Down: [intermediate_size, hidden_size]
        vec![0.0; intermediate_size as usize];
    ];
    
    result
}

pub fn llama_m_grad(input_dim: usize, hidden_dim: usize) -> std::sync::Arc<std::sync::Mutex<Vec<f32>>> {
    let result = Arc::new(std::sync::LazyLock::new(vec![0.0]));
    result.clone()
}
