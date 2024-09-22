use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Sigmoid;

impl Sigmoid {
    pub fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Matrix) -> Matrix {
        input.map(|x| Self::activate(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_sigmoid_success() {
        assert_eq!(Sigmoid::activate(0.0), 0.5);
        assert_eq!(Sigmoid::activate(0.5), 0.6224593312018546);
    }
}