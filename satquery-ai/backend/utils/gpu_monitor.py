"""
SatQuery AI — GPU Memory Monitoring
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("satquery.gpu")


class GPUMonitor:
    """Monitor GPU memory usage and availability."""
    
    def __init__(self):
        self.has_gpu = False
        self.gpu_info: Dict = {}
        self._initialize()
    
    def _initialize(self):
        """Initialize GPU monitoring capabilities."""
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            
            if self.has_gpu:
                device_count = torch.cuda.device_count()
                self.gpu_info = {
                    "available": True,
                    "device_count": device_count,
                    "devices": [],
                }
                
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    device_props = torch.cuda.get_device_properties(i)
                    
                    device_info = {
                        "index": i,
                        "name": device_name,
                        "total_memory_mb": device_props.total_memory / (1024 * 1024),
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "multi_processor_count": device_props.multi_processor_count,
                    }
                    self.gpu_info["devices"].append(device_info)
                
                logger.info(f"GPU monitoring initialized: {device_count} device(s) available")
            else:
                self.gpu_info = {"available": False}
                logger.info("GPU monitoring initialized: No GPU available")
                
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
            self.has_gpu = False
            self.gpu_info = {"available": False}
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitoring: {e}")
            self.has_gpu = False
            self.gpu_info = {"available": False, "error": str(e)}
    
    def get_current_memory(self, device_id: int = 0) -> Optional[Dict]:
        """Get current GPU memory usage for a specific device."""
        if not self.has_gpu:
            return None
        
        try:
            import torch
            
            if device_id >= torch.cuda.device_count():
                logger.warning(f"Device {device_id} not available")
                return None
            
            # Switch to device to get memory info
            with torch.cuda.device(device_id):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                max_allocated = torch.cuda.max_memory_allocated(device_id)
                max_reserved = torch.cuda.max_memory_reserved(device_id)
                
                device_props = torch.cuda.get_device_properties(device_id)
                total_memory = device_props.total_memory
                
                return {
                    "allocated_mb": allocated / (1024 * 1024),
                    "reserved_mb": reserved / (1024 * 1024),
                    "total_mb": total_memory / (1024 * 1024),
                    "free_mb": (total_memory - allocated) / (1024 * 1024),
                    "utilization_percent": (allocated / total_memory) * 100 if total_memory > 0 else 0,
                    "max_allocated_mb": max_allocated / (1024 * 1024),
                    "max_reserved_mb": max_reserved / (1024 * 1024),
                }
                
        except Exception as e:
            logger.error(f"Failed to get GPU memory for device {device_id}: {e}")
            return None
    
    def get_all_memory_stats(self) -> Dict:
        """Get memory stats for all available GPUs."""
        if not self.has_gpu:
            return {"available": False, "device_count": 0, "devices": []}
        
        stats = {
            "available": True,
            "device_count": self.gpu_info.get("device_count", 0),
            "devices": [],
        }
        
        for device_info in self.gpu_info.get("devices", []):
            device_id = device_info["index"]
            memory_stats = self.get_current_memory(device_id)
            
            if memory_stats:
                device_stats = {
                    **device_info,
                    **memory_stats,
                }
                stats["devices"].append(device_stats)
        
        return stats
    
    def estimate_model_memory(self, model_name: str, batch_size: int = 1) -> Tuple[float, float]:
        """
        Estimate memory requirements for a model.
        Returns: (estimated_memory_mb, safe_threshold_mb)
        """
        # Rough estimates based on model type and size
        model_estimates = {
            # BLIP models
            "Salesforce/blip-vqa-base": (1500, 2000),
            "Salesforce/blip2-opt-2.7b": (4000, 5000),
            "Salesforce/blip-image-captioning-base": (1500, 2000),
            
            # CLIP models
            "ViT-B-32": (500, 750),
            "ViT-L-14": (1000, 1500),
            
            # OWL-ViT
            "google/owlvit-base-patch32": (1000, 1500),
            
            # ResNet
            "microsoft/resnet-50": (500, 750),
        }
        
        # Find best matching estimate
        for key, (base_mem, safe_mem) in model_estimates.items():
            if key in model_name:
                # Adjust for batch size (rough linear scaling)
                adjusted_base = base_mem * batch_size
                adjusted_safe = safe_mem * batch_size
                return adjusted_base, adjusted_safe
        
        # Default estimates based on model size indicators
        if "blip2" in model_name.lower() or "2.7b" in model_name:
            return 4000 * batch_size, 5000 * batch_size
        elif "blip" in model_name.lower():
            return 1500 * batch_size, 2000 * batch_size
        elif "vit-l" in model_name.lower():
            return 1000 * batch_size, 1500 * batch_size
        elif "vit-b" in model_name.lower():
            return 500 * batch_size, 750 * batch_size
        else:
            # Conservative default
            return 1000 * batch_size, 1500 * batch_size
    
    def check_memory_sufficient(self, model_name: str, batch_size: int = 1, 
                                device_id: int = 0) -> Tuple[bool, str, Dict]:
        """
        Check if there's sufficient GPU memory to load a model.
        Returns: (is_sufficient, message, stats)
        """
        if not self.has_gpu:
            return True, "No GPU available, using CPU", {}
        
        memory_stats = self.get_current_memory(device_id)
        if not memory_stats:
            return False, "Failed to get GPU memory stats", {}
        
        estimated_mem, safe_mem = self.estimate_model_memory(model_name, batch_size)
        free_memory = memory_stats["free_mb"]
        
        stats = {
            "estimated_required_mb": estimated_mem,
            "safe_threshold_mb": safe_mem,
            "current_free_mb": free_memory,
            "current_allocated_mb": memory_stats["allocated_mb"],
            "total_memory_mb": memory_stats["total_mb"],
        }
        
        if free_memory >= safe_mem:
            return True, f"Sufficient memory: {free_memory:.0f}MB free, {estimated_mem:.0f}MB estimated", stats
        elif free_memory >= estimated_mem:
            return True, f"Barely sufficient memory: {free_memory:.0f}MB free, {estimated_mem:.0f}MB estimated (close to limit)", stats
        else:
            return False, f"Insufficient memory: {free_memory:.0f}MB free, {estimated_mem:.0f}MB estimated required", stats
    
    def clear_cache(self, device_id: int = 0):
        """Clear GPU cache to free memory."""
        if not self.has_gpu:
            return False
        
        try:
            import torch
            
            if device_id >= torch.cuda.device_count():
                return False
            
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device_id)
                logger.info(f"Cleared GPU cache for device {device_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")
            return False


# Singleton instance
_gpu_monitor: Optional[GPUMonitor] = None

def get_gpu_monitor() -> GPUMonitor:
    """Get or create the GPU monitor singleton."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor