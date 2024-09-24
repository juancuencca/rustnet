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
}