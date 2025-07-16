#!/usr/bin/env python3
"""
AMASS数据集探索工具
用于高效查看retarget到G1的AMASS数据集的内容、标签和统计信息
"""

import numpy as np
import os
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional

class AMASSDataExplorer:
    """AMASS数据集探索器"""
    
    def __init__(self, data_root: str = "g1"):
        """
        初始化数据探索器
        
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
        self.datasets = {}
        self.stats = {}
        
    def scan_datasets(self) -> Dict[str, Dict]:
        """
        扫描所有数据集并收集基本信息
        
        Returns:
            包含所有数据集信息的字典
        """
        print("正在扫描数据集...")
        
        for dataset_dir in self.data_root.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                self.datasets[dataset_name] = {
                    'path': dataset_dir,
                    'subjects': {},
                    'total_files': 0,
                    'total_frames': 0,
                    'file_sizes': []
                }
                
                # 扫描子目录（动作序列）
                for subject_dir in dataset_dir.iterdir():
                    if subject_dir.is_dir():
                        subject_name = subject_dir.name
                        files = list(subject_dir.glob("*.npz"))
                        
                        if files:
                            self.datasets[dataset_name]['subjects'][subject_name] = {
                                'path': subject_dir,
                                'files': files,
                                'file_count': len(files),
                                'frames': [],
                                'durations': [],
                                'file_sizes': []
                            }
                            
                            # 分析每个文件
                            for file_path in files:
                                try:
                                    data = np.load(file_path)
                                    fps = data['fps'].item()
                                    num_frames = data['body_positions'].shape[0]
                                    duration = num_frames / fps
                                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                                    
                                    self.datasets[dataset_name]['subjects'][subject_name]['frames'].append(num_frames)
                                    self.datasets[dataset_name]['subjects'][subject_name]['durations'].append(duration)
                                    self.datasets[dataset_name]['subjects'][subject_name]['file_sizes'].append(file_size)
                                    
                                    self.datasets[dataset_name]['total_files'] += 1
                                    self.datasets[dataset_name]['total_frames'] += num_frames
                                    self.datasets[dataset_name]['file_sizes'].append(file_size)
                                    
                                except Exception as e:
                                    print(f"警告: 无法读取文件 {file_path}: {e}")
        
        return self.datasets
    
    def print_summary(self):
        """打印数据集摘要信息"""
        print("\n" + "="*60)
        print("AMASS数据集摘要")
        print("="*60)
        
        total_datasets = len(self.datasets)
        total_files = sum(ds['total_files'] for ds in self.datasets.values())
        total_frames = sum(ds['total_frames'] for ds in self.datasets.values())
        
        print(f"数据集数量: {total_datasets}")
        print(f"总文件数: {total_files}")
        print(f"总帧数: {total_frames:,}")
        print(f"估计总时长: {total_frames/120:.1f} 秒 ({total_frames/120/60:.1f} 分钟)")
        
        print("\n各数据集统计:")
        print("-" * 60)
        print(f"{'数据集':<20} {'文件数':<8} {'帧数':<12} {'平均时长(s)':<12}")
        print("-" * 60)
        
        for name, info in sorted(self.datasets.items()):
            avg_duration = info['total_frames'] / 120 if info['total_files'] > 0 else 0
            print(f"{name:<20} {info['total_files']:<8} {info['total_frames']:<12,} {avg_duration:<12.1f}")
    
    def explore_dataset(self, dataset_name: str, max_subjects: int = 5):
        """
        探索特定数据集的详细信息
        
        Args:
            dataset_name: 数据集名称
            max_subjects: 最大显示的子目录数量
        """
        if dataset_name not in self.datasets:
            print(f"错误: 数据集 '{dataset_name}' 不存在")
            return
        
        dataset = self.datasets[dataset_name]
        print(f"\n数据集: {dataset_name}")
        print("="*50)
        
        subjects = list(dataset['subjects'].items())[:max_subjects]
        
        for subject_name, subject_info in subjects:
            print(f"\n子目录: {subject_name}")
            print(f"  文件数: {subject_info['file_count']}")
            print(f"  总帧数: {sum(subject_info['frames']):,}")
            print(f"  平均时长: {np.mean(subject_info['durations']):.2f}s")
            print(f"  平均文件大小: {np.mean(subject_info['file_sizes']):.2f}MB")
            
            # 显示前几个文件名
            files = subject_info['files'][:3]
            for file_path in files:
                filename = file_path.name
                print(f"    - {filename}")
            
            if len(subject_info['files']) > 3:
                print(f"    ... 还有 {len(subject_info['files']) - 3} 个文件")
    
    def analyze_motion_file(self, file_path: str):
        """
        分析单个动作文件的详细内容
        
        Args:
            file_path: 文件路径
        """
        try:
            data = np.load(file_path)
            print(f"\n文件分析: {file_path}")
            print("="*50)
            
            # 基本信息
            fps = data['fps'].item()
            body_positions = data['body_positions']
            body_rotations = data['body_rotations']
            dof_positions = data['dof_positions']
            
            print(f"FPS: {fps}")
            print(f"帧数: {body_positions.shape[0]}")
            print(f"时长: {body_positions.shape[0]/fps:.2f}秒")
            print(f"身体部位数: {body_positions.shape[1]}")
            print(f"关节数: {dof_positions.shape[1]}")
            
            # 数据形状
            print(f"\n数据形状:")
            print(f"  body_positions: {body_positions.shape}")
            print(f"  body_rotations: {body_rotations.shape}")
            print(f"  dof_positions: {dof_positions.shape}")
            
            # 统计信息
            print(f"\n统计信息:")
            print(f"  位置范围: X[{body_positions[:,:,0].min():.3f}, {body_positions[:,:,0].max():.3f}]")
            print(f"            Y[{body_positions[:,:,1].min():.3f}, {body_positions[:,:,1].max():.3f}]")
            print(f"            Z[{body_positions[:,:,2].min():.3f}, {body_positions[:,:,2].max():.3f}]")
            print(f"  关节角度范围: [{dof_positions.min():.3f}, {dof_positions.max():.3f}]")
            
            # 检查数据质量
            print(f"\n数据质量检查:")
            has_nan = np.isnan(body_positions).any() or np.isnan(dof_positions).any()
            has_inf = np.isinf(body_positions).any() or np.isinf(dof_positions).any()
            print(f"  包含NaN: {has_nan}")
            print(f"  包含Inf: {has_inf}")
            
        except Exception as e:
            print(f"错误: 无法分析文件 {file_path}: {e}")
    
    def find_similar_motions(self, dataset_name: str, duration_range: Tuple[float, float] = None):
        """
        查找特定时长的动作
        
        Args:
            dataset_name: 数据集名称
            duration_range: 时长范围 (min, max)
        """
        if dataset_name not in self.datasets:
            print(f"错误: 数据集 '{dataset_name}' 不存在")
            return
        
        dataset = self.datasets[dataset_name]
        motions = []
        
        for subject_name, subject_info in dataset['subjects'].items():
            for i, duration in enumerate(subject_info['durations']):
                if duration_range is None or (duration_range[0] <= duration <= duration_range[1]):
                    motions.append({
                        'subject': subject_name,
                        'file': subject_info['files'][i].name,
                        'duration': duration,
                        'frames': subject_info['frames'][i],
                        'file_size': subject_info['file_sizes'][i]
                    })
        
        if motions:
            print(f"\n找到 {len(motions)} 个符合条件的动作:")
            print("-" * 80)
            print(f"{'子目录':<15} {'文件名':<25} {'时长(s)':<8} {'帧数':<8} {'大小(MB)':<8}")
            print("-" * 80)
            
            for motion in sorted(motions, key=lambda x: x['duration']):
                print(f"{motion['subject']:<15} {motion['file']:<25} {motion['duration']:<8.2f} "
                      f"{motion['frames']:<8} {motion['file_size']:<8.2f}")
        else:
            print("未找到符合条件的动作")
    
    def generate_statistics_plot(self, save_path: str = None):
        """
        生成数据集统计图表
        
        Args:
            save_path: 保存路径（可选）
        """
        # 准备数据
        dataset_names = list(self.datasets.keys())
        file_counts = [self.datasets[name]['total_files'] for name in dataset_names]
        frame_counts = [self.datasets[name]['total_frames'] for name in dataset_names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 文件数量分布
        ax1.bar(dataset_names, file_counts)
        ax1.set_title('各数据集文件数量')
        ax1.set_xlabel('数据集')
        ax1.set_ylabel('文件数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 帧数分布
        ax2.bar(dataset_names, frame_counts)
        ax2.set_title('各数据集总帧数')
        ax2.set_xlabel('数据集')
        ax2.set_ylabel('帧数')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"统计图表已保存到: {save_path}")
        else:
            plt.show()
    
    def export_summary_csv(self, output_path: str = "amass_summary.csv"):
        """
        导出数据集摘要到CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        data = []
        
        for dataset_name, dataset_info in self.datasets.items():
            for subject_name, subject_info in dataset_info['subjects'].items():
                for i, file_path in enumerate(subject_info['files']):
                    data.append({
                        'dataset': dataset_name,
                        'subject': subject_name,
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'frames': subject_info['frames'][i],
                        'duration': subject_info['durations'][i],
                        'file_size_mb': subject_info['file_sizes'][i],
                        'fps': 120  # 假设所有文件都是120fps
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"数据集摘要已导出到: {output_path}")
        return df

def main():
    parser = argparse.ArgumentParser(description="AMASS数据集探索工具")
    parser.add_argument("--data_root", default="g1", help="数据根目录")
    parser.add_argument("--action", choices=["summary", "explore", "analyze", "find", "plot", "export"], 
                       default="summary", help="执行的操作")
    parser.add_argument("--dataset", help="要探索的数据集名称")
    parser.add_argument("--file", help="要分析的文件路径")
    parser.add_argument("--duration_min", type=float, help="最小时长（秒）")
    parser.add_argument("--duration_max", type=float, help="最大时长（秒）")
    parser.add_argument("--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 初始化探索器
    explorer = AMASSDataExplorer(args.data_root)
    
    # 扫描数据集
    explorer.scan_datasets()
    
    if args.action == "summary":
        explorer.print_summary()
        
    elif args.action == "explore":
        if not args.dataset:
            print("错误: 需要指定 --dataset 参数")
            return
        explorer.explore_dataset(args.dataset)
        
    elif args.action == "analyze":
        if not args.file:
            print("错误: 需要指定 --file 参数")
            return
        explorer.analyze_motion_file(args.file)
        
    elif args.action == "find":
        if not args.dataset:
            print("错误: 需要指定 --dataset 参数")
            return
        duration_range = None
        if args.duration_min is not None or args.duration_max is not None:
            duration_range = (args.duration_min or 0, args.duration_max or float('inf'))
        explorer.find_similar_motions(args.dataset, duration_range)
        
    elif args.action == "plot":
        explorer.generate_statistics_plot(args.output)
        
    elif args.action == "export":
        explorer.export_summary_csv(args.output or "amass_summary.csv")

if __name__ == "__main__":
    main() 