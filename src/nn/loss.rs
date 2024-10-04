use crate::matrix::Matrix;
use std::f64::EPSILON; 

pub trait Loss {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64;
    fn gradient(&self) -> Matrix;
}

pub struct MSELoss {
    cached_target: Option<Matrix>,
    cached_predicted: Option<Matrix>,
}

impl MSELoss {
    pub fn new() -> MSELoss {
        MSELoss { 
            cached_target: None, 
            cached_predicted: None 
        }
    }

    fn loss(target: f64, predicted: f64) -> f64 {
        (target - predicted).powi(2)
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        2.0 * (predicted - target) 
    }
}

impl Loss for MSELoss {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64 {
        self.cached_target = Some(target.clone());
        self.cached_predicted = Some(predicted.clone());

        let n = target.values.len() as f64;        
        
        let values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::loss(target, predicted))
            .collect::<Vec<f64>>();

        let sum = values.iter().fold(0.0, |acc, x| acc + x);
        
        sum / n
    }

    fn gradient(&self) -> Matrix {
        let target = self.cached_target.as_ref().expect("No cached target found. Did you forget to call compute()?");
        let predicted = self.cached_predicted.as_ref().expect("No cached predicted found. Did you forget to call compute()?");

        let n = target.values.len() as f64;
        
        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::derivative(target, predicted) / n)
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len(), 1), grad_values)
    }
}

pub struct MAELoss {
    cached_target: Option<Matrix>,
    cached_predicted: Option<Matrix>,
}

impl MAELoss {
    pub fn new() -> MAELoss {
        MAELoss { 
            cached_target: None, 
            cached_predicted: None 
        }
    }

    fn loss(target: f64, predicted: f64) -> f64 {
        (target - predicted).abs()
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        if predicted > target { 1.0 } 
        else if predicted < target { -1.0 } 
        else { 0.0 } 
    }
}

impl Loss for MAELoss {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64 {
        self.cached_target = Some(target.clone());
        self.cached_predicted = Some(predicted.clone());

        let n = target.values.len() as f64;        
        
        let values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::loss(target, predicted))
            .collect::<Vec<f64>>();

        let sum = values.iter().fold(0.0, |acc, x| acc + x);
        
        sum / n
    }

    fn gradient(&self) -> Matrix {
        let target = self.cached_target.as_ref().expect("No cached target found. Did you forget to call compute()?");
        let predicted = self.cached_predicted.as_ref().expect("No cached predicted found. Did you forget to call compute()?");

        let n = target.values.len() as f64;
        
        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::derivative(target, predicted) / n)
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len(), 1), grad_values)
    }
}

pub struct BCELoss {
    cached_target: Option<Matrix>,
    cached_predicted: Option<Matrix>,
}

impl BCELoss {
    pub fn new() -> BCELoss {
        BCELoss { 
            cached_target: None, 
            cached_predicted: None 
        }
    }

    fn loss(target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);

        -target * predicted.ln() - (1.0 - target) * (1.0 - predicted).ln()
    }

    fn derivative(target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);
    
        -target / predicted + (1.0 - target) / (1.0 - predicted)
    }
}

impl Loss for BCELoss {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64 {
        self.cached_target = Some(target.clone());
        self.cached_predicted = Some(predicted.clone());

        let n = target.values.len() as f64;        
        
        let values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::loss(target, predicted))
            .collect::<Vec<f64>>();

        let sum = values.iter().fold(0.0, |acc, x| acc + x);
        
        sum / n
    }

    fn gradient(&self) -> Matrix {
        let target = self.cached_target.as_ref().expect("No cached target found. Did you forget to call compute()?");
        let predicted = self.cached_predicted.as_ref().expect("No cached predicted found. Did you forget to call compute()?");

        let n = target.values.len() as f64;
        
        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| Self::derivative(target, predicted) / n)
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len(), 1), grad_values)
    }
}

pub struct AUCReshaping {
    n: f64,
    thetamax: f64,
    cached_target: Option<Matrix>,
    cached_predicted: Option<Matrix>,
}

impl AUCReshaping {
    pub fn new(n: f64, thetamax: f64) -> AUCReshaping {
        AUCReshaping { 
            n, 
            thetamax, 
            cached_target: None, 
            cached_predicted: None 
        }
    }

    fn loss(&self, target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);

        let b_i = if target == 1.0 && predicted < self.thetamax {
            self.n
        } else {
            0.0
        };

        -target * (predicted - b_i).ln() - (1.0 - target) * (1.0 - predicted + b_i).ln()
    }

    fn derivative(&self, target: f64, predicted: f64) -> f64 {
        let predicted = predicted.clamp(EPSILON, 1.0 - EPSILON);
    
        let b_i = if target == 1.0 && predicted < self.thetamax {
            self.n
        } else {
            0.0
        };
    
        -target / (predicted - b_i) + (1.0 - target) / (1.0 - predicted + b_i)
    }
}

impl Loss for AUCReshaping {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64 {
        self.cached_target = Some(target.clone());
        self.cached_predicted = Some(predicted.clone());

        let n = target.values.len() as f64;        
        
        let values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| self.loss(target, predicted))
            .collect::<Vec<f64>>();

        let sum = values.iter().fold(0.0, |acc, x| acc + x);
        
        sum / n
    }

    fn gradient(&self) -> Matrix {
        let target = self.cached_target.as_ref().expect("No cached target found. Did you forget to call compute()?");
        let predicted = self.cached_predicted.as_ref().expect("No cached predicted found. Did you forget to call compute()?");

        let n = target.values.len() as f64;
        
        let grad_values = target.values
            .iter()
            .zip(predicted.values.iter())
            .map(|(&target, &predicted)| self.derivative(target, predicted) / n)
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len(), 1), grad_values)
    }
}

pub struct CrossEntropy {
    cached_target: Option<Matrix>,
    cached_predicted: Option<Matrix>,
}

impl CrossEntropy {
    pub fn new() -> CrossEntropy {
        CrossEntropy { cached_target: None, cached_predicted: None }
    }

    fn loss(target: &[f64], predicted: &[f64]) -> f64 {
        target
            .iter()
            .zip(predicted.iter())
            .map(|(&t, &p)| {
                let p = p.clamp(std::f64::EPSILON, 1.0 - std::f64::EPSILON);
                -t * p.ln()
            })
            .sum()
    }

    fn derivative(target: &[f64], predicted: &[f64]) -> Vec<f64> {
        target
            .iter()
            .zip(predicted.iter())
            .map(|(t, p)| p - t)
            .collect()
    }
}

impl Loss for CrossEntropy {
    fn compute(&mut self, target: &Matrix, predicted: &Matrix) -> f64 {
        self.cached_target = Some(target.clone());
        self.cached_predicted = Some(predicted.clone());

        let n = target.values.len() as f64;

        let values = target.values
            .chunks(target.cols) 
            .zip(predicted.values.chunks(predicted.cols))
            .map(|(y_true_row, y_pred_row)| Self::loss(y_true_row, y_pred_row))
            .collect::<Vec<f64>>();

        let sum = values.iter().fold(0.0, |acc, x| acc + x);

        sum / n
    }

    fn gradient(&self) -> Matrix {
        let target = self.cached_target.as_ref().expect("No cached target found. Did you forget to call compute()?");
        let predicted = self.cached_predicted.as_ref().expect("No cached predicted found. Did you forget to call compute()?");

        let n = target.values.len() as f64;

        let grad_values = target.values
            .chunks(target.cols)
            .zip(predicted.values.chunks(predicted.cols))
            .flat_map(|(y_true_row, y_pred_row)| {
                Self::derivative(y_true_row, y_pred_row)
                    .iter()
                    .map(|grad| grad / n)
                    .collect::<Vec<f64>>() 
            })
            .collect::<Vec<f64>>();

        Matrix::new((grad_values.len() / target.cols, target.cols), grad_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bce_loss_success() {
        let target = Matrix::new((3, 1), vec![1.0, 0.0, 1.0]);
        let predicted = Matrix::new((3, 1), vec![0.8, 0.2, 0.7]);
        
        let mut bce_loss = BCELoss::new();
        let loss = bce_loss.compute(&target, &predicted);

        assert_eq!(loss, 0.26765401552238394);
    }
}