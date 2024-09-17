use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Linear {
    weights: Matrix,
    biases: Matrix 
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::rand((input_size, output_size));
        let biases = Matrix::zeros((1, output_size));
        Linear { weights, biases }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        input.dot(&self.weights)
            .unwrap()
            .sum(&self.biases)
            .unwrap()
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
    fn test_forward_success() {
        let linear = Linear::new(2, 2);
        let input_matrix = Matrix::new((1, 2), vec![0.5, 0.5]).unwrap();
        let result = linear.forward(&input_matrix);

        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 2);
        assert_eq!(result.values, vec![0.58151125407333445, 0.4743134840669604]);    
    }
}
