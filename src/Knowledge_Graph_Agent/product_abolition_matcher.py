"""
产品废止日期匹配器
用于根据文档内容匹配产品废止日期
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional


class ProductAbolitionMatcher:
    """产品废止日期匹配器"""
    
    def __init__(self, abolition_schedule_path: str):
        """
        初始化匹配器
        
        Args:
            abolition_schedule_path: 产品废止时间表文件路径
        """
        self.product_abolition_map = self._load_abolition_schedule(abolition_schedule_path)
    
    def _load_abolition_schedule(self, file_path: str) -> Dict[str, str]:
        """
        加载产品废止时间表
        
        Args:
            file_path: 文件路径
            
        Returns:
            产品名称到废止日期的映射字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持数组格式，转换为字典
            if isinstance(data, list):
                abolition_map = {}
                for item in data:
                    product_name = item.get("ProductName", "")
                    sales_stop_time = item.get("SalesStopTime", "")
                    if product_name and sales_stop_time:
                        abolition_map[product_name] = sales_stop_time
                return abolition_map
            else:
                # 原字典格式直接返回
                return data
        except Exception as e:
            print(f"加载产品废止时间表失败: {e}")
            return {}
    
    def match_product_abolition_date(self, content: str) -> Optional[str]:
        """
        根据内容匹配产品废止日期
        
        Args:
            content: 文档内容
            
        Returns:
            匹配到的废止日期，未匹配到返回None
        """
        if not content or not self.product_abolition_map:
            return None
        
        # 将内容转换为小写以提高匹配准确性
        content_lower = content.lower()
        
        # 遍历所有产品名称，查找匹配
        for product_name, abolition_date in self.product_abolition_map.items():
            # 使用简单的字符串匹配，可以根据需要调整匹配策略
            if product_name.lower() in content_lower:
                return abolition_date
        
        return None


# 全局匹配器实例
_abolition_matcher = None


def get_abolition_matcher(abolition_schedule_path: Optional[str] = None) -> ProductAbolitionMatcher:
    """
    获取全局产品废止日期匹配器实例
    
    Args:
        abolition_schedule_path: 产品废止时间表文件路径，首次调用时必须提供
        
    Returns:
        ProductAbolitionMatcher实例
    """
    global _abolition_matcher
    
    if _abolition_matcher is None and abolition_schedule_path:
        _abolition_matcher = ProductAbolitionMatcher(abolition_schedule_path)
    
    return _abolition_matcher


def match_product_abolition_date(content: str) -> Optional[str]:
    """
    根据内容匹配产品废止日期的便捷函数
    
    Args:
        content: 文档内容
        
    Returns:
        匹配到的废止日期，未匹配到返回None
    """
    global _abolition_matcher
    
    if _abolition_matcher is None:
        return None
    
    return _abolition_matcher.match_product_abolition_date(content)