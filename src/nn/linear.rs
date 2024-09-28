use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Linear {
    weights: Matrix,
    biases: Matrix, 
    cached_input: Option<Matrix>,
    learning_rate: f64,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, learning_rate: f64, (l_range, r_range): (f64, f64), seed: Option<u64>) -> Linear {
        let weights = Matrix::rand((input_size, output_size), (l_range, r_range), seed);
        let biases = Matrix::zeros((1, output_size));

        Linear { 
            weights, 
            biases, 
            cached_input: None,
            learning_rate
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.cached_input = Some(input.clone());
        
        input
            .dot(&self.weights)
            .add(&self.biases)
    }

    pub fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self.cached_input.as_ref().expect("No cached input found. Did you forget to call forward()?");

        let grad_weights = input.transpose().dot(&grad_output);
        let grad_biases = grad_output.sum_rows();
        let grad_input = grad_output.dot(&self.weights.transpose());

        self.weights = self.weights.sub(&grad_weights.scale(self.learning_rate));
        self.biases = self.biases.sub(&grad_biases.scale(self.learning_rate));

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
        let seed = 42;
        let mut linear = Linear::new(2, 2, Some(seed));
        let input = Matrix::new((1, 2), vec![
            0.5, 0.5
        ]);
        let result = linear.forward(&input);

        assert_eq!((result.rows, result.cols), (1, 2));
        assert_eq!(result.values, vec![
            0.58151125407333445, 0.4743134840669604
        ]);    
    }
}
