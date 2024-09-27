pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let correct = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == x2).count();

    correct as f64 / y_true.len() as f64
}

pub fn precision(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let true_positive = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 1.0 && x2 == 1.0).count();
    let false_positive = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 0.0 && x2 == 1.0).count();

    true_positive  as f64 / (true_positive + false_positive) as f64
}

// sensitivity = recall = true positive rate
pub fn sensitivity(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let true_positive = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 1.0 && x2 == 1.0).count();
    let false_negative = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 1.0 && x2 == 0.0).count();

    true_positive  as f64 / (true_positive + false_negative) as f64
}

pub fn specificity(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let true_negative = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 0.0 && x2 == 0.0).count();
    let false_positive = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 0.0 && x2 == 1.0).count();

    true_negative as f64 / (true_negative + false_positive) as f64
}

// false positive rate = 1 - specificity
pub fn false_positive_rate(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let true_negative = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 0.0 && x2 == 0.0).count();
    let false_positive = y_true.iter().zip(y_pred.iter()).filter(|(&x1, &x2)| x1 == 0.0 && x2 == 1.0).count();

    1.0 - true_negative as f64 / (true_negative + false_positive) as f64
}

pub fn f1_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let precision = precision(y_true, y_pred);
    let recall = sensitivity(y_true, y_pred);

    2.0 * (precision * recall) / (precision + recall)
}

pub fn fbeta_score(y_true: &[f64], y_pred: &[f64], beta: f64) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "array's length mismatch");

    let precision = precision(y_true, y_pred);
    let recall = sensitivity(y_true, y_pred);

    (1.0 + beta) * (precision * recall) / (beta * precision + recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(accuracy(&y_true, &y_pred), 0.7);
    }

    #[test]
    fn test_precision_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(precision(&y_true, &y_pred), 0.6666666666666666);
    }

    #[test]
    fn test_sensitivity_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(sensitivity(&y_true, &y_pred), 0.8);
    }

    #[test]
    fn test_specificity_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(specificity(&y_true, &y_pred), 0.6);
    }

    #[test]
    fn test_false_positive_rate_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(false_positive_rate(&y_true, &y_pred), 0.4);
    }

    #[test]
    fn test_f1_score_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(f1_score(&y_true, &y_pred), 0.7272727272727272);
    }

    #[test]
    fn test_fbeta_score_success() {
        let y_true = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];

        assert_eq!(fbeta_score(&y_true, &y_pred, 0.5), 0.7058823529411765);
    }
}