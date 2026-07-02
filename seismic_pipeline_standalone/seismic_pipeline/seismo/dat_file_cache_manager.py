"""
Dat file cache manager for EEG data files.

This module provides caching functionality for .dat files from both
local/network drives and S3 storage, with focus on hypnogram calculation.
"""

import os
import glob
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
import logging
import boto3
from datetime import datetime
import tempfile

from seismic_pipeline.config.paths import local_data_root as default_local_data_root


class DatFileCacheManagerYt:
    """
    Specialized cache manager for .dat files from multiple sources.
    
    This class handles caching of .dat files from both local/network drives
    and S3 storage, with focus on hypnogram calculation from cached data.
    """
    
    def __init__(self, 
                 local_cache_dir: str = './dat_file_cache',
                 local_data_root: str | None = None,
                 s3_config: Optional[Dict] = None,
                 s3_rat_bucket: str = 'rat'):
        """
        Initialize the dat file cache manager.
        
        Parameters
        ----------
        local_cache_dir : str, default='./dat_file_cache'
            Local directory for caching .dat files
        local_data_root : str, default='/mnt/wd/rat'
            Root directory for local/network drive data
        s3_config : dict, optional
            S3 connection configuration
        s3_rat_bucket : str, default='rat'
            S3 bucket name for rat data
        """
        self.local_cache_dir = Path(local_cache_dir)
        self.local_data_root = Path(
            local_data_root if local_data_root is not None else default_local_data_root()
        )
        self.s3_config = s3_config
        self.s3_rat_bucket = s3_rat_bucket
        
        # Create cache directory
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 client handling (similar to HypnogramCacheManagerYt)
        self._s3_client = None
        
        # Class-level cache: {frozenset(config_items): boto3_client}
        if not hasattr(DatFileCacheManagerYt, "_CLIENT_CACHE"):
            DatFileCacheManagerYt._CLIENT_CACHE = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache index for tracking cached files
        self.cache_index_file = self.local_cache_dir / 'cache_index.pkl'
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        import pickle
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        return {}
        
    def _save_cache_index(self):
        """Save cache index to disk."""
        import pickle
        try:
            with open(self.cache_index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, rat_id: str, date: str) -> str:
        """Generate a unique cache key for rat and date."""
        # Normalize date format to ensure consistent cache keys
        date_normalized = self._parse_date(date)
        return f"{rat_id}_{date_normalized}"
        
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to YYYY_MM_DD format."""
        try:
            # Handle different date formats
            if '_' in date_str:
                return date_str
            elif '-' in date_str:
                return date_str.replace('-', '_')
            else:
                # Assume YYYYMMDD format
                return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
        except (IndexError, ValueError):
            return date_str
    
    def _get_rat_dat_list_local(self, date: str, rat_id: str) -> List[str]:
        """Find all .dat files for given day containing the rat ID in the filename."""
        date_parsed = self._parse_date(date)
        search_dir = self.local_data_root / date_parsed
        if not search_dir.is_dir():
            return []
        paths = glob.glob(str(search_dir / "**" / "*.dat"), recursive=True)
        return sorted([p for p in paths if rat_id in os.path.basename(p)])
    
    def _load_dat_from_local(self, rat_id: str, date: str) -> Optional[np.ndarray]:
        """Load and concatenate all .dat files for a day/rat from local filesystem."""
        try:
            files = self._get_rat_dat_list_local(date, rat_id)
            if not files:
                return None
            
            data = np.empty(shape=(0, 4))
            for fp in files:
                raw = np.fromfile(file=fp, dtype=np.int16)
                # Prefer 8-channel layout; fallback to 4-channel if needed
                arr = None
                if raw.size % 8 == 0:
                    try:
                        arr = raw.reshape((-1, 8))[:, :4]
                    except Exception:
                        arr = None
                if arr is None:
                    # Fallback to 4 channels if 8 doesn't fit
                    arr = raw.reshape((-1, 4))
                data = np.append(data, arr, axis=0)
            
            return data if data.size > 0 else None
            
        except Exception as e:
            self.logger.error(f"Failed to load .dat from local: {e}")
            return None
    
    def _get_rat_dat_list_s3(self, date: str, rat_id: str) -> List[str]:
        """List .dat keys in 'rat' bucket for a given day containing the rat ID."""
        client = self._get_s3_client()
        if client is None:
            return []
        
        try:
            date_parsed = self._parse_date(date)
            keys = []
            paginator = client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.s3_rat_bucket, Prefix=f"{date_parsed}/"):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if (rat_id in key) and key.endswith(".dat"):
                            keys.append(key)
            return sorted(keys)
        except Exception as e:
            self.logger.error(f"Failed to list S3 .dat files: {e}")
            return []
    
    def _load_dat_from_s3(self, rat_id: str, date: str) -> Optional[np.ndarray]:
        """Download .dat files from S3 and load them."""
        client = self._get_s3_client()
        if client is None:
            self.logger.error("S3 client not initialized")
            return None
        
        try:
            keys = self._get_rat_dat_list_s3(date, rat_id)
            if not keys:
                return None
            
            data = np.empty(shape=(0, 4))
            for key in keys:
                # Download to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp_file:
                    client.download_file(self.s3_rat_bucket, key, tmp_file.name)
                    
                    raw = np.fromfile(file=tmp_file.name, dtype=np.int16)
                    os.unlink(tmp_file.name)
                    
                    # Prefer 8-channel layout; fallback to 4-channel if needed
                    arr = None
                    if raw.size % 8 == 0:
                        try:
                            arr = raw.reshape((-1, 8))[:, :4]
                        except Exception:
                            arr = None
                    if arr is None:
                        # Fallback to 4 channels if 8 doesn't fit
                        arr = raw.reshape((-1, 4))
                    data = np.append(data, arr, axis=0)
            
            return data if data.size > 0 else None
            
        except Exception as e:
            self.logger.error(f"Failed to load .dat from S3: {e}")
            return None
    
    def cache_dat_file(self, rat_id: str, date: str, source: str = 'local') -> bool:
        """
        Cache .dat file for a specific rat and date.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'R1', 'R2', etc.)
        date : str
            Date in YYYY_MM_DD format
        source : str, default='local'
            Source to load from ('local' or 's3')
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(rat_id, date)
            
            # Check if already cached
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                if cache_file.exists():
                    try:
                        # Verify cache file is readable
                        test_data = np.fromfile(cache_file, dtype=np.int16)
                        if test_data.size > 0:
                            return True
                    except Exception:
                        # Cache file corrupted, remove it
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                        del self.cache_index[cache_key]
                        self._save_cache_index()
            
            # Load .dat file from source
            if source == 'local':
                data = self._load_dat_from_local(rat_id, date)
            elif source == 's3':
                data = self._load_dat_from_s3(rat_id, date)
            else:
                self.logger.error(f"Unknown source: {source}")
                return False
            
            if data is None or data.size == 0:
                return False
            
            # Save to cache
            cache_file = self.local_cache_dir / f"{cache_key}_dat.npy"
            np.save(cache_file, data)
            
            # Update cache index
            self.cache_index[cache_key] = {
                'rat_id': rat_id,
                'date': date,
                'source': source,
                'cache_file': str(cache_file),
                'cached_at': datetime.now().isoformat(),
                'data_shape': data.shape
            }
            self._save_cache_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache .dat file: {e}")
            return False
    
    def get_cached_dat_file(self, rat_id: str, date: str, source: str = 'local') -> Optional[np.ndarray]:
        """
        Get cached .dat file for a specific rat and date.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier
        date : str
            Date in YYYY_MM_DD format
        source : str, default='local'
            Source to try first ('local' or 's3')
            
        Returns
        -------
        np.ndarray or None
            Cached .dat data if found, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(rat_id, date)
            
            # Check cache first
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                
                if cache_file.exists():
                    try:
                        data = np.load(cache_file)
                        return data
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached .dat file: {e}")
                        # Remove corrupted cache entry
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                        del self.cache_index[cache_key]
                        self._save_cache_index()
            
            # Not cached, try to cache it
            if source == 'local':
                success = self.cache_dat_file(rat_id, date, 'local')
                if success:
                    return self.get_cached_dat_file(rat_id, date, 'local')
                # Try S3 as fallback
                if self.s3_config:
                    success = self.cache_dat_file(rat_id, date, 's3')
                    if success:
                        return self.get_cached_dat_file(rat_id, date, 's3')
            elif source == 's3' and self.s3_config:
                success = self.cache_dat_file(rat_id, date, 's3')
                if success:
                    return self.get_cached_dat_file(rat_id, date, 's3')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached .dat file: {e}")
            return None
    
    def _get_s3_client(self):
        """Lazily create or fetch a boto3 client for the current configuration."""
        if self._s3_client is not None:
            return self._s3_client
        if not self.s3_config:
            return None
        
        # Ensure class attribute exists
        if not hasattr(DatFileCacheManagerYt, "_CLIENT_CACHE"):
            DatFileCacheManagerYt._CLIENT_CACHE = {}
        
        cfg_key = frozenset(self.s3_config.items())
        if cfg_key in DatFileCacheManagerYt._CLIENT_CACHE:
            self._s3_client = DatFileCacheManagerYt._CLIENT_CACHE[cfg_key]
        else:
            # Separate service_name from other kwargs for boto3.client
            cfg = dict(self.s3_config)
            service_name = cfg.pop('service_name', 's3')
            self._s3_client = boto3.client(service_name, **cfg)
            DatFileCacheManagerYt._CLIENT_CACHE[cfg_key] = self._s3_client
        return self._s3_client
    
    # Make the class picklable by excluding the boto3 client
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_s3_client'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._s3_client = None



