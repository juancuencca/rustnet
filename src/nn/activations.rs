use crate::matrix::Matrix;
use super::sequential::Layer;

pub struct ReLU {
    cached_input: Option<Matrix>,
}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU { 
            cached_input: None 
        }
    }

    pub fn activate(x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { x }  
    }

    pub fn relu_derivative(x: f64) -> f64 {
        if x == 0.0 { 0.0 } else { 1.0 }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.cached_input = Some(input.clone());

        input.map(|x| Self::activate(x))
    }

    pub fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self.cached_input.as_ref().expect("No cached input found. Did you forget to call forward()?");
        let grad_input = input.map(|x| Self::relu_derivative(x));
        
        grad_input.hadamard_product(grad_output)
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        self.backward(grad_output)
    }
}

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

    fn sigmoid_derivative(output: f64) -> f64 {
        output * (1.0 - output)
    }

    pub fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let output = self.cached_output.as_ref().expect("No cached output found. Did you forget to call forward()?");
        let grad_input = output.map(|o| Self::sigmoid_derivative(o));
        
        grad_input.hadamard_product(grad_output)
    }
}

impl Layer for Sigmoid {
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
    fn test_activation_sigmoid_success() {
        assert_eq!(Sigmoid::activate(0.0), 0.5);
        assert_eq!(Sigmoid::activate(0.5), 0.6224593312018546);
    }
}