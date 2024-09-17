use crate::matrix::Matrix;

pub trait Layer {
    fn forward(&self, input: &Matrix) -> Matrix;
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    } 

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.layers.iter()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn predict(&self, train_x: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {
        let mut predicted_values = Vec::new();
        for row in train_x {
            let x = Matrix::new((1, row.len()), row.clone())?;
            let a = self.forward(&x);

            predicted_values.push(a.values[0]);
        }
    
        Ok(predicted_values)
    }

}