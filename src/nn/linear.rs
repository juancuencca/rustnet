use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Linear {
    weights: Matrix,
    biases: Matrix, 
    cached_input: Option<Matrix>,
    lr: f64,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, l_rng: f64, r_rng: f64, lr: f64, seed: Option<u64>) -> Linear {
        let weights = Matrix::rand(input_size, output_size, l_rng, r_rng, seed);
        let biases = Matrix::zeros(1, output_size);

        Linear { weights, biases, cached_input: None, lr, }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.cached_input = Some(input.clone());
        
        input.multiply(&self.weights).add_bias(&self.biases)
    }

    pub fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self.cached_input.as_ref().expect("No cached input found. Did you forget to call forward()?");

        let grad_weights = input.transpose().multiply(&grad_output);
        let grad_biases = grad_output.sum_rows();
        let grad_input = grad_output.multiply(&self.weights.transpose());

        self.weights = self.weights.sub(&grad_weights.scale(self.lr));
        self.biases = self.biases.sub(&grad_biases.scale(self.lr));

        grad_input
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        self.backward(grad_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_success() {
        let mut linear = Linear::new(2, 2, 0.0, 1.0, 1.0, None);
        let input = Matrix::new(1, 2, vec![
            0.5, 0.5
        ]);
        let result = linear.forward(&input);

        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 2);
    }
}
