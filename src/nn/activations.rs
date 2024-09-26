use crate::matrix::Matrix;
use super::sequential::Layer;

pub trait ActivationFunction {
    fn activate(x: f64) -> f64;
    fn derivative(x: f64) -> f64;

    fn apply(input: &Matrix) -> Matrix {
        input.map(|x| Self::activate(x))
    }

    fn compute_gradient(input: &Matrix, grad_output: &Matrix) -> Matrix {
        let grad_input = input.map(|x| Self::derivative(x));
        
        grad_input.hadamard_product(grad_output)
    }
}

pub struct Relu {
    cached_input: Option<Matrix>,
}

impl Relu {
    pub fn new() -> Relu {
        Relu { 
            cached_input: None 
        }
    }
}

impl ActivationFunction for Relu {
    fn activate(x: f64) -> f64 {
        if x > 0.0 { 
            x 
        } else { 
            0.0 
        }  
    }

    fn derivative(x: f64) -> f64 {
        if x > 0.0 { 
            1.0 
        } else { 
            0.0 
        }
    }
}

impl Layer for Relu {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.cached_input = Some(input.clone());

        Self::apply(input)
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self.cached_input.as_ref().expect("No cached input found. Did you forget to call forward()?");

        Self::compute_gradient(input, grad_output)
    }
}

pub struct Sigmoid {
    cached_input: Option<Matrix>
}

impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {
            cached_input: None
        }
    }
}
 
impl ActivationFunction for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    fn derivative(x: f64) -> f64 {
        let s = Self::activate(x);
        
        s * (1.0 - s)
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.cached_input = Some(input.clone());

        Self::apply(input)
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self.cached_input.as_ref().expect("No cached input found. Did you forget to call forward()?");

        Self::compute_gradient(input, grad_output)
    }
}

pub struct Softmax {
    cached_output: Option<Matrix>, 
}

impl Softmax {
    pub fn new() -> Softmax {
        Softmax { 
            cached_output: None 
        }
    }

    fn activate(logits: &[f64]) -> Vec<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp_values: f64 = exp_values.iter().sum();
        
        exp_values.iter().map(|&x| x / sum_exp_values).collect()
    }

    pub fn apply(input: &Matrix) -> Matrix {
        input.map_chunks(input.cols, |chunk| Self::activate(chunk))
    }

    fn softmax_derivative(softmax_output: &[f64]) -> Vec<f64> {
        let size = softmax_output.len();
        let mut jacobian = vec![0.0; size * size]; 

        for i in 0..size {
            for j in 0..size {
                if i == j {
                    jacobian[i * size + j] = softmax_output[i] * (1.0 - softmax_output[i]);
                } else {
                    jacobian[i * size + j] = -softmax_output[i] * softmax_output[j];
                }
            }
        }

        jacobian
    }

    pub fn compute_gradient(&self, grad_output: &Matrix) -> Matrix {
        let cached_output = self.cached_output.as_ref().expect("No cached output found. Did you forget to call forward()?");

        let mut new_gradients = Vec::with_capacity(grad_output.values.len());

        for (chunk_softmax, chunk_grad_output) in cached_output.values.chunks(cached_output.cols).zip(grad_output.values.chunks(grad_output.cols)) {
            let jacobian = Self::softmax_derivative(chunk_softmax); 

            for i in 0..chunk_grad_output.len() {
                let mut grad = 0.0;
                for j in 0..chunk_grad_output.len() {
                    grad += jacobian[i * chunk_grad_output.len() + j] * chunk_grad_output[j];
                }
                new_gradients.push(grad);
            }
        }

        Matrix {
            rows: grad_output.rows,
            cols: grad_output.cols,
            values: new_gradients,
        }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let softmax_output = Self::apply(input);
        self.cached_output = Some(softmax_output.clone());

        softmax_output
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        self.compute_gradient(grad_output)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_success() {
        assert_eq!(Sigmoid::activate(0.0), 0.5);
    }

    #[test]
    fn test_sigmoid_derivative_success() {
        assert_eq!(Sigmoid::derivative(0.0), 0.25);   
    }

    #[test]
    fn test_relu_success() {
        assert_eq!(Relu::activate(3.0), 3.0);
        assert_eq!(Relu::activate(-1.0), 0.0);
    }

    #[test]
    fn test_relu_derivative_success() {
        assert_eq!(Relu::derivative(3.0), 1.0);
        assert_eq!(Relu::derivative(-1.0), 0.0);   
    }

    #[test]
    fn test_softmax_success() {
        assert_eq!(Softmax::activate(&vec![-1.0, 0.0, 3.0, 5.0]), vec![0.002165696460061088, 0.005886973333342136, 0.11824302025266467, 0.8737043099539322]);
    }
}