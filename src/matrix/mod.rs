use rand::{Rng, SeedableRng};
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    pub fn new((rows, cols): (usize, usize), values: Vec<f64>) -> Matrix {
        assert_eq!(rows * cols, values.len(), "Error: mismatch array length with shape provided");
        
        Matrix { 
            rows, 
            cols, 
            values, 
        }
    } 

    pub fn rand((rows, cols): (usize, usize), seed: Option<u64>) -> Matrix {
        let mut rng: Box<dyn RngCore> = match seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::thread_rng()),
        };

        let values = (0..(rows * cols))
            .into_iter()
            .map(|_| rng.gen_range(0.0..=1.0))
            .collect::<Vec<f64>>(); 
        
        Matrix { 
            rows, 
            cols, 
            values 
        }
    }

    pub fn zeros((rows, cols): (usize, usize)) -> Matrix {
        let values = (0..(rows * cols))
            .into_iter()
            .map(|_| 0.0)
            .collect::<Vec<f64>>();

        Matrix { 
            rows, 
            cols, 
            values 
        }
    }
}

impl Matrix {
    pub fn map<F>(&self, f: F) -> Matrix
    where  
        F: Fn(f64) -> f64 
    {
        let values = self.values
            .iter()
            .map(|&x| f(x))
            .collect::<Vec<f64>>();

        Matrix { 
            rows: self.rows, 
            cols: self.cols, 
            values 
        }
    } 

    pub fn map_chunks<F>(&self, chunk_size: usize, f: F) -> Matrix
    where  
        F: Fn(&[f64]) -> Vec<f64> 
    {
        let mut new_values = Vec::with_capacity(self.values.len());
        
        for chunk in self.values.chunks(chunk_size) {
            let transformed_chunk = f(chunk); 
            new_values.extend(transformed_chunk); 
        }

        Matrix { 
            rows: self.rows, 
            cols: self.cols, 
            values: new_values 
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut values = Vec::new();
        
        for j in 0..self.cols {
            for i in 0..self.rows {
                values.push(self.get_value(i, j));
            }
        }

        Matrix { 
            rows: self.cols, 
            cols: self.rows, 
            values 
        }
    }

    pub fn dot(&self, m: &Matrix) -> Matrix {
        assert_eq!(self.cols, m.rows, "Error: matrices shape mismatch");

        let mut values = Vec::new();

        for i in 0..self.rows {
            for j in 0..m.cols {
                let mut sum = 0.0;
                
                for k in 0..self.cols {
                    sum += self.get_value(i, k) * m.get_value(k, j);        
                }

                values.push(sum);
            }
        }

        Matrix { 
            rows: self.rows, 
            cols: m.cols, 
            values 
        }
    }

    pub fn add(&self, m: &Matrix) -> Matrix {
        assert_eq!(self.cols, m.cols, "Matrix columns must match.");
        assert_eq!(m.rows, 1, "Parameter's matrix column must be 1");

        let mut values = vec![0.0; self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                values[i * self.cols + j] = self.get_value(i, j) + m.get_value(0, j); 
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            values,
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix dimensions must match.");
        assert_eq!(self.cols, other.cols, "Matrix dimensions must match.");

        let values = self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix { 
            rows: self.rows, 
            cols: self.cols, 
            values 
        }
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let values = self.values
            .iter()
            .map(|&val| val * scalar)
            .collect();

        Matrix { 
            rows: self.rows, 
            cols: self.cols, 
            values 
        }
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut sums = vec![0.0; self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                sums[col] += self.get_value(row, col);
            }
        }

        Matrix { 
            rows: 1, 
            cols: self.cols, 
            values: sums 
        }
    }

    pub fn hadamard_product(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrices must have the same dimensions for element-wise multiplication.");
        assert_eq!(self.cols, other.cols, "Matrices must have the same dimensions for element-wise multiplication.");

        let values = self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<f64>>();

        Matrix { 
            rows: self.rows, 
            cols: self.cols, 
            values 
        }
    }

    fn get_value(&self, row: usize, col: usize) -> f64 {
        self.values[(row * self.cols) + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_success() {
        let matrix = Matrix::new((1, 2), vec![0.0, 1.0]);

        assert_eq!((matrix.rows, matrix.cols), (1, 2));
        assert_eq!(matrix.values, vec![
            0.0, 1.0
        ]);
    }

    #[test]
    fn test_matrix_rand_creation_success() {
        let seed = 42;
        
        let f_matrix = Matrix::rand((2, 2), Some(seed));
        assert_eq!((f_matrix.rows, f_matrix.cols), (2, 2));
        assert_eq!(f_matrix.values, vec![
            0.5265574090027739, 0.542725209903144, 
            0.636465099143895, 0.4059017582307768
        ]);
    
        let s_matrix = Matrix::rand((2, 1), Some(seed));
        assert_eq!((s_matrix.rows, s_matrix.cols), (2, 1));
        assert_eq!(s_matrix.values, vec![
            0.5265574090027739, 
            0.542725209903144
        ]);
    }

    #[test]
    fn test_matrix_zeros_creation_success() {
        let matrix = Matrix::zeros((2, 2));

        assert_eq!((matrix.rows, matrix.cols), (2, 2));
        assert_eq!(matrix.values, vec![
            0.0, 0.0,
            0.0, 0.0
        ]);
    }
    
    #[test]
    fn test_matrix_transposed_success() {
        let matrix = Matrix::new((2, 3), vec![
            2.0, 3.0, 4.0, 
            5.0, 6.0, 7.0
        ]);

        let transposed = matrix.transpose();

        assert_eq!((transposed.rows, transposed.cols), (3, 2));
        assert_eq!(transposed.values, vec![
            2.0, 5.0, 
            3.0, 6.0, 
            4.0, 7.0
        ]);
    }

    #[test]
    fn test_matrix_dot_success() {
        let f_matrix = Matrix::new((2, 1), vec![
            2.0, 
            3.0
        ]);
        let s_matrix = Matrix::new((1, 2), vec![
            2.0, 3.0
        ]);

        let r_matrix = f_matrix.dot(&s_matrix);

        assert_eq!((r_matrix.rows, r_matrix.cols), (2, 2));
        assert_eq!(r_matrix.values, vec![
            4.0, 6.0, 
            6.0, 9.0
        ]);
    }

    #[test]
    fn test_matrix_addition_success() {
        let f_matrix = Matrix::new((4, 1), vec![
            2.0, 
            3.0,
            4.0, 
            5.0,
        ]);
        
        let s_matrix = Matrix::new((1, 1), vec![
            2.0, 
        ]);

        let r_matrix = f_matrix.add(&s_matrix);
        assert_eq!((r_matrix.rows, r_matrix.cols), (4, 1));
        assert_eq!(r_matrix.values, vec![
            4.0, 
            5.0,
            6.0, 
            7.0,
        ]);
    }

    #[test]
    fn test_matrix_sum_rows_success() {
        let f_matrix = Matrix::new((1, 3), vec![
            1.0, 2.0, 3.0
        ]);
        let f_summed = f_matrix.sum_rows(); 
        assert_eq!((f_summed.rows, f_summed.cols), (1, 3)); 
        assert_eq!(f_summed.values, vec![
            1.0, 2.0, 3.0
        ]);
    
        let s_matrix = Matrix::new((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let s_summed = s_matrix.sum_rows(); 
        assert_eq!((s_summed.rows, s_summed.cols), (1, 2)); 
        assert_eq!(s_summed.values, vec![
            4.0, 6.0
        ]);
    }

    #[test]
    fn test_matrix_substraction_success() {
        let f_matrix = Matrix::new((1, 3), vec![
            4.0, 6.0, 5.0
        ]);
        let s_matrix = Matrix::new((1, 3), vec![
            1.0, 2.0, 1.0
        ]);

        let r_sub = f_matrix.sub(&s_matrix);

        assert_eq!((r_sub.rows, r_sub.cols), (1, 3)); 
        assert_eq!(r_sub.values, vec![
            3.0, 4.0, 4.0
        ]);
    }

    #[test]
    fn test_matrix_hadamard_product_success() {
        let f_matrix = Matrix::new((1, 3), vec![
            4.0, 6.0, 5.0
        ]);
        let s_matrix = Matrix::new((1, 3), vec![
            1.0, 2.0, 1.0
        ]);

        let r_prod = f_matrix.hadamard_product(&s_matrix);

        assert_eq!((r_prod.rows, r_prod.cols), (1, 3)); 
        assert_eq!(r_prod.values, vec![
            4.0, 12.0, 5.0
        ]);
    }

    #[test]
    fn test_matrix_scale_success() {
        let f_matrix = Matrix::new((1, 3), vec![
            4.0, 6.0, 5.0
        ]);
        
        let r_scaled = f_matrix.scale(2.0);

        assert_eq!((r_scaled.rows, r_scaled.cols), (1, 3)); 
        assert_eq!(r_scaled.values, vec![
            8.0, 12.0, 10.0
        ]);
    }
}
