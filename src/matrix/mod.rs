use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const RANDOM_SEED: u64 = 42;

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    pub fn new((rows, cols): (usize, usize), values: Vec<f64>) -> Result<Self, &'static str> {
        if rows * cols != values.len() {
            return Err("Err: missmatch array length with shape provided");
        }

        Ok(Self { rows, cols, values })
    } 

    pub fn rand((rows, cols): (usize, usize)) -> Self {
        let mut rng = StdRng::seed_from_u64(RANDOM_SEED);

        let values = (0..(rows * cols)).into_iter()
            .map(|_| rng.gen_range(0.0..=1.0))
            .collect::<Vec<f64>>(); 

        Self { rows, cols, values }
    }

    pub fn zeros((rows, cols): (usize, usize)) -> Self {
        let values = (0..(rows * cols)).into_iter()
            .map(|_| 0.0)
            .collect::<Vec<f64>>();
        
        Self { rows, cols, values }
    }
    
    pub fn map<F>(&self, f: F) -> Self
    where  
        F: Fn(f64) -> f64 
    {
        let values = self.values.iter()
            .map(|&x| f(x))
            .collect::<Vec<f64>>();
        Self { rows: self.rows, cols: self.cols, values }
    } 

    pub fn transpose(&self, ) -> Self {
        let mut transposed = Vec::new();
        
        for j in 0..self.cols {
            for i in 0..self.rows {
                transposed.push(self.get_value(i, j));
            }
        }

        Self { rows: self.cols, cols: self.rows, values: transposed }
    }

    pub fn dot(&self, t: &Self) -> Result<Self, &'static str> {
        if self.cols != t.rows {
            return Err("Error: matrices shape missmatch");
        }

        let mut values = Vec::new();

        for i in 0..self.rows {
            for j in 0..t.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get_value(i, k) * t.get_value(k, j);        
                }
                values.push(sum);
            }
        }

        Ok(Self { rows: self.rows, cols: t.cols, values })
    }

    pub fn sum(&self, t: &Self) -> Result<Self, &'static str> {
        if self.rows != t.rows || self.cols != t.cols {
            return Err("Error: matrices shape missmatch");
        }

        let mut values = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                values.push(self.get_value(i, j) + t.get_value(i, j));
            }
        }

        Ok(Self { rows: self.rows, cols: self.cols, values })
    }

    fn get_value(&self, row: usize, col: usize) -> f64 {
        self.values[(row * self.cols) + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_random_matrix_success() {
        let mat = Matrix::rand((2, 2));
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 2);
        assert_eq!(mat.values, vec![0.5265574090027739, 0.542725209903144, 0.636465099143895, 0.4059017582307768]);
    
        let mat = Matrix::rand((2, 1));
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 1);
        assert_eq!(mat.values, vec![0.5265574090027739, 0.542725209903144]);
    }

    #[test]
    fn test_create_zeros_matrix_success() {
        assert_eq!(Matrix::zeros((2, 2)), Matrix {
            rows: 2, cols: 2, values: vec![0.0, 0.0, 0.0, 0.0]
        });
    }
    
    #[test]
    fn test_transpose_matrix_success() {
        let mat = Matrix { rows: 2, cols: 3, values: vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]};
        assert_eq!(mat.transpose(), Matrix {
            rows: 3, cols: 2, values: vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]
        });
    }

    #[test]
    fn test_dot_matrix_error_mismatch_shape() {
        let m = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };
        let t = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };

        let r = m.dot(&t);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), "Error: matrices shape missmatch");
    }

    #[test]
    fn test_dot_matrix_success() {
        let m = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };
        let t = Matrix { rows: 1, cols: 2, values: vec![2.0, 3.0] };

        let r = m.dot(&t);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), Matrix { rows: 2, cols: 2, values: vec![4.0, 6.0, 6.0, 9.0]});
    }

    #[test]
    fn test_sum_matrix_error_mismatch_shape() {
        let m = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };
        let t = Matrix { rows: 2, cols: 2, values: vec![2.0, 3.0, 2.0, 2.0] };

        let r = m.sum(&t);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err(), "Error: matrices shape missmatch");
    }

    #[test]
    fn test_sum_matrix_success() {
        let m = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };
        let t = Matrix { rows: 2, cols: 1, values: vec![2.0, 3.0] };

        let r = m.sum(&t);
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), Matrix { rows: 2, cols: 1, values: vec![4.0, 6.0] });
    }
}
