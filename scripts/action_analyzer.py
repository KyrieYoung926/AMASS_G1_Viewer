#!/usr/bin/env python3
"""
AMASS数据集动作类型分析工具
用于快速了解数据集中包含的动作类型
"""

import numpy as np
import os
import argparse
from pathlib import Path
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set

class ActionAnalyzer:
    """动作类型分析器"""
    
    def __init__(self, data_root: str = "g1"):
        self.data_root = Path(data_root)
        self.action_patterns = {
            # 舞蹈动作
            'dance': ['dance', 'flamenco', 'salsa', 'bachata', 'reggaeton', 'karsilamas', 'zeimpekiko', 'hasapiko', 'maleviziotikos', 'haniotikos'],
            # 情感动作
            'emotion': ['happy', 'sad', 'angry', 'tired', 'excited', 'afraid', 'annoyed', 'nervous', 'scary', 'satisfied'],
            # 日常动作
            'daily': ['walk', 'run', 'jump', 'sit', 'stand', 'reach', 'grab', 'throw', 'catch'],
            # 运动动作
            'sport': ['basketball', 'football', 'tennis', 'swimming', 'boxing', 'kick', 'punch'],
            # 手势动作
            'gesture': ['wave', 'point', 'clap', 'shake', 'nod', 'bow'],
            # 其他
            'other': ['mix', 'musical', 'theater', 'performance']
        }
    
    def analyze_dataset_actions(self, dataset_name: str):
        """分析特定数据集的动作类型"""
        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            print(f"❌ 数据集 '{dataset_name}' 不存在")
            return
        
        print(f"🎭 分析数据集: {dataset_name}")
        print("="*60)
        
        # 收集所有文件名
        all_files = []
        for subject in dataset_path.iterdir():
            if subject.is_dir():
                files = list(subject.glob("*.npz"))
                all_files.extend(files)
        
        print(f"📁 总文件数: {len(all_files)}")
        
        # 分析文件名中的动作信息
        action_info = self._extract_action_info(all_files)
        
        # 显示动作类型统计
        self._show_action_statistics(action_info)
        
        # 显示具体动作示例
        self._show_action_examples(action_info)
    
    def _extract_action_info(self, files: List[Path]) -> Dict:
        """从文件名中提取动作信息"""
        action_info = {
            'categories': defaultdict(list),
            'specific_actions': defaultdict(list),
            'file_count': len(files)
        }
        
        for file_path in files:
            filename = file_path.name.lower()
            
            # 按类别分类
            categorized = False
            for category, patterns in self.action_patterns.items():
                for pattern in patterns:
                    if pattern in filename:
                        action_info['categories'][category].append(file_path)
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                action_info['categories']['unknown'].append(file_path)
            
            # 提取具体动作名称
            action_name = self._extract_action_name(filename)
            if action_name:
                action_info['specific_actions'][action_name].append(file_path)
        
        return action_info
    
    def _extract_action_name(self, filename: str) -> str:
        """从文件名中提取动作名称"""
        # 移除常见的后缀
        filename = filename.replace('_poses_120_jpos.npz', '').replace('_poses_60_jpos.npz', '')
        filename = filename.replace('_c3d_poses_120_jpos.npz', '').replace('_c3d_poses_60_jpos.npz', '')
        
        # 提取动作部分（通常在最后一个下划线后）
        parts = filename.split('_')
        if len(parts) >= 2:
            # 尝试提取动作名称
            potential_action = parts[-1]
            if potential_action.isdigit():
                # 如果最后是数字，尝试前面的部分
                if len(parts) >= 3:
                    potential_action = parts[-2]
            
            return potential_action
        
        return filename
    
    def _show_action_statistics(self, action_info: Dict):
        """显示动作类型统计"""
        print(f"\n📊 动作类型统计:")
        print("-" * 40)
        
        total_files = action_info['file_count']
        for category, files in action_info['categories'].items():
            if files:
                percentage = len(files) / total_files * 100
                print(f"{category.capitalize():<12}: {len(files):<4} 文件 ({percentage:.1f}%)")
    
    def _show_action_examples(self, action_info: Dict):
        """显示具体动作示例"""
        print(f"\n🎯 具体动作示例:")
        print("-" * 40)
        
        # 显示每个类别的示例
        for category, files in action_info['categories'].items():
            if files and category != 'unknown':
                print(f"\n{category.capitalize()} 动作:")
                examples = files[:3]  # 显示前3个示例
                for file_path in examples:
                    action_name = self._extract_action_name(file_path.name)
                    print(f"  - {action_name} ({file_path.name})")
                if len(files) > 3:
                    print(f"  ... 还有 {len(files) - 3} 个文件")
        
        # 显示未知类别的文件
        if action_info['categories']['unknown']:
            print(f"\n未知类别文件:")
            examples = action_info['categories']['unknown'][:5]
            for file_path in examples:
                print(f"  - {file_path.name}")
            if len(action_info['categories']['unknown']) > 5:
                print(f"  ... 还有 {len(action_info['categories']['unknown']) - 5} 个文件")
    
    def list_all_datasets_actions(self):
        """列出所有数据集的动作类型概览"""
        datasets = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        print("🎭 所有数据集动作类型概览")
        print("="*80)
        
        for dataset in sorted(datasets):
            dataset_path = self.data_root / dataset
            files = list(dataset_path.rglob("*.npz"))
            
            if files:
                # 快速分析动作类型
                action_types = set()
                for file_path in files[:10]:  # 只分析前10个文件
                    filename = file_path.name.lower()
                    for category, patterns in self.action_patterns.items():
                        for pattern in patterns:
                            if pattern in filename:
                                action_types.add(category)
                                break
                
                print(f"{dataset:<20} {len(files):<6} 文件  {', '.join(sorted(action_types)) if action_types else '未知'}")
    
    def find_specific_action(self, action_keyword: str, dataset_name: str = None):
        """查找包含特定动作的文件"""
        print(f"🔍 查找包含 '{action_keyword}' 的动作文件")
        print("="*60)
        
        found_files = []
        
        if dataset_name:
            datasets = [dataset_name]
        else:
            datasets = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        for dataset in datasets:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                continue
            
            for file_path in dataset_path.rglob("*.npz"):
                if action_keyword.lower() in file_path.name.lower():
                    found_files.append((dataset, file_path))
        
        if found_files:
            print(f"找到 {len(found_files)} 个文件:")
            print("-" * 80)
            print(f"{'数据集':<15} {'文件名':<50} {'路径'}")
            print("-" * 80)
            
            for dataset, file_path in found_files[:20]:  # 只显示前20个
                print(f"{dataset:<15} {file_path.name:<50} {file_path}")
            
            if len(found_files) > 20:
                print(f"... 还有 {len(found_files) - 20} 个文件")
        else:
            print(f"❌ 未找到包含 '{action_keyword}' 的文件")

def main():
    parser = argparse.ArgumentParser(description="AMASS数据集动作类型分析工具")
    parser.add_argument("--action", choices=["analyze", "list", "find"], 
                       default="list", help="执行的操作")
    parser.add_argument("--dataset", help="要分析的数据集名称")
    parser.add_argument("--keyword", help="要查找的动作关键词")
    
    args = parser.parse_args()
    
    analyzer = ActionAnalyzer()
    
    if args.action == "analyze":
        if not args.dataset:
            print("❌ 需要指定 --dataset 参数")
            return
        analyzer.analyze_dataset_actions(args.dataset)
    
    elif args.action == "list":
        analyzer.list_all_datasets_actions()
    
    elif args.action == "find":
        if not args.keyword:
            print("❌ 需要指定 --keyword 参数")
            return
        analyzer.find_specific_action(args.keyword, args.dataset)

if __name__ == "__main__":
    main() 