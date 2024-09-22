use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&self, input: &Matrix) -> Matrix;
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

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.layers
            .iter()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }
}