use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, grad_output: &Matrix) -> Matrix;
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { 
            layers: vec![] 
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    } 

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn backward(&mut self, loss_grad: &Matrix) {
        let mut grad = loss_grad.clone();

        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }
}