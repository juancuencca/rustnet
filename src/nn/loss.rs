pub struct MeanSquaredError;

impl MeanSquaredError {
    fn calculate(target: f64, predicted: f64) -> f64 {
        (target - predicted).powi(2)
    }

    pub fn apply_to_list(target_values: &[f64], predicted_values: &[f64]) -> f64 {
        let n = predicted_values.len();
        let values = (0..n).into_iter()
            .map(|i| Self::calculate(predicted_values[i], target_values[i]))
            .collect::<Vec<f64>>();

        values.iter().fold(0.0, |acc, x| acc + (x / values.len() as f64))
    }
}