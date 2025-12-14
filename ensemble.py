"""
Ensemble methods for MABe Mouse Behavior Detection
"""
from typing import List, Tuple, Optional
import numpy as np

from config import Config


class ImprovedEnsemble:
    """Weighted ensemble with confidence-based voting"""
    
    @staticmethod
    def weighted_average(
        predictions: List[np.ndarray], 
        thresholds: List[float],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Weighted average of predictions from multiple folds
        
        Args:
            predictions: List of probability arrays
            thresholds: List of corresponding thresholds
            weights: Optional weights for each fold
            
        Returns:
            Weighted averaged predictions
        """
        if not predictions:
            return np.array([])
        
        n_folds = len(predictions)
        
        if weights is None:
            if Config.USE_WEIGHTED_ENSEMBLE:
                # Exponential decay weights
                weights = [Config.FOLD_WEIGHT_DECAY ** i for i in range(n_folds)]
            else:
                weights = [1.0] * n_folds
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Apply threshold and calculate masked predictions
        masked_preds = []
        for pred, thresh, weight in zip(predictions, thresholds, weights):
            labels = (pred >= thresh).astype(np.float32)
            masked = pred * labels * weight
            masked_preds.append(masked)
        
        return np.sum(masked_preds, axis=0)
    
    @staticmethod
    def confidence_voting(
        predictions: List[np.ndarray],
        thresholds: List[float],
        min_agreement: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voting mechanism with confidence scores
        
        Args:
            predictions: List of probability arrays
            thresholds: List of thresholds
            min_agreement: Minimum ratio of folds that must agree
            
        Returns:
            Tuple of (final_predictions, confidence_scores)
        """
        n_folds = len(predictions)
        n_samples = len(predictions[0])
        
        # Count positive votes from folds
        votes = np.zeros(n_samples)
        prob_sum = np.zeros(n_samples)
        
        for pred, thresh in zip(predictions, thresholds):
            positive = (pred >= thresh).astype(np.float32)
            votes += positive
            prob_sum += pred * positive
        
        # Agreement ratio
        agreement = votes / n_folds
        
        # Only keep predictions with enough agreement
        final_mask = agreement >= min_agreement
        
        # Confidence = average probability when positive
        confidence = np.where(votes > 0, prob_sum / np.maximum(votes, 1), 0)
        
        # Final prediction
        final_pred = np.where(final_mask, confidence, 0)
        
        return final_pred, confidence

