"""
Metadata adder for preserving event information in the final dataset.

This module provides functionality to add event metadata (date, rat_id) to the final
feature matrix while keeping it separate from the actual features used by the model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
from ..mod.sklearnbaseyt import TransformerMixinYt


class MetadataAdderYt(TransformerMixinYt):
    """
    Adds event metadata (original event date, rat_id) to the final dataset.
    The metadata is added as additional columns but should not be used for training.
    """
    
    def __init__(self, metadata_columns: List[str] = None):
        """
        Initialize the metadata adder.
        
        Parameters
        ----------
        metadata_columns : list of str, optional
            List of metadata column names to add. If None, uses default columns.
        """
        self.metadata_columns = metadata_columns or ['original_event_date', 'original_rat_id']
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Add metadata to the feature matrix.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,), optional
            Target values
            
        Returns
        -------
        X_with_metadata : array-like of shape (n_samples, n_features + n_metadata)
            Feature matrix with metadata columns added
        y : array-like of shape (n_samples,)
            Target values (unchanged)
        """
        if not hasattr(X, 'shape') or len(X.shape) != 2:
            self.logger.warning("X is not a 2D array, cannot add metadata")
            return X, y
        
        # Get metadata from the previous step (REMProfileCalculatorYt)
        if hasattr(self, '_metadata') and self._metadata:
            metadata_array = self._create_metadata_array(self._metadata)
            
            # Combine features with metadata
            X_with_metadata = np.hstack([X, metadata_array])
            
            self.logger.info(f"Added {metadata_array.shape[1]} metadata columns to features")
            return X_with_metadata, y
        else:
            self.logger.warning("No metadata available, returning original X")
            return X, y
    
    def _create_metadata_array(self, metadata: List[Dict]) -> np.ndarray:
        """
        Create a metadata array from metadata list.
        
        Parameters
        ----------
        metadata : list of dict
            List of metadata dictionaries
            
        Returns
        -------
        metadata_array : array-like of shape (n_samples, n_metadata_columns)
            Metadata as numerical array
        """
        metadata_data = []
        
        for meta in metadata:
            row = []
            for col in self.metadata_columns:
                value = meta.get(col, '')
                # Convert to numerical representation
                if col == 'original_event_date':
                    # Convert date to numerical format (YYYYMMDD)
                    try:
                        if '_' in value:
                            date_parts = value.split('_')
                            if len(date_parts) == 3:
                                numerical_date = int(date_parts[0]) * 10000 + int(date_parts[1]) * 100 + int(date_parts[2])
                            else:
                                numerical_date = 0
                        else:
                            numerical_date = 0
                    except:
                        numerical_date = 0
                    row.append(numerical_date)
                elif col == 'original_rat_id':
                    # Convert rat_id to numerical (R1=1, R2=2, etc.)
                    try:
                        if value.startswith('R'):
                            rat_num = int(value[1:])
                        else:
                            rat_num = 0
                    except:
                        rat_num = 0
                    row.append(rat_num)
                else:
                    row.append(0)  # Default value for unknown columns
            
            metadata_data.append(row)
        
        return np.array(metadata_data)
    
    def set_metadata(self, metadata: List[Dict]):
        """
        Set metadata from the previous transformer.
        
        Parameters
        ----------
        metadata : list of dict
            List of metadata dictionaries
        """
        self._metadata = metadata

