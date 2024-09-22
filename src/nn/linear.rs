use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Linear {
    weights: Matrix,
    biases: Matrix, 
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, seed: Option<u64>) -> Linear {
        let weights = Matrix::rand((input_size, output_size), seed);
        let biases = Matrix::zeros((1, output_size));

        Linear { 
            weights, 
            biases, 
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        input
            .dot(&self.weights)
            .add(&self.biases)
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Matrix) -> Matrix {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_success() {
        let seed = 42;
        let linear = Linear::new(2, 2, Some(seed));
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
