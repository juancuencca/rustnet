use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct Sigmoid {
    cached_output: Option<Matrix>
}

impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {
            cached_output: None
        }
    }
    
    pub fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn forward(&mut self, input: &Matrix) -> Matrix {
        let output = input.map(|x| Self::activate(x));
        self.cached_output = Some(output.clone());
        
        output
    }

}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
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