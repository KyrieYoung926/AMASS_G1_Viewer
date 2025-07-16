#!/usr/bin/env python3
"""
AMASS数据集快速查看工具
提供快速的数据浏览和预览功能
"""

import numpy as np
import os
import argparse
from pathlib import Path
import random
from typing import List, Tuple

def quick_stats(data_root: str = "g1"):
    """快速统计数据集基本信息"""
    data_path = Path(data_root)
    datasets = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    print(f"📁 发现 {len(datasets)} 个数据集:")
    for ds in sorted(datasets):
        print(f"   - {ds}")
    
    # 随机选择几个文件进行快速分析
    print(f"\n🔍 随机采样分析...")
    sample_files = []
    for dataset in random.sample(datasets, min(3, len(datasets))):
        dataset_path = data_path / dataset
        subjects = [s for s in dataset_path.iterdir() if s.is_dir()]
        if subjects:
            subject = random.choice(subjects)
            files = list(subject.glob("*.npz"))
            if files:
                sample_files.append(random.choice(files))
    
    total_frames = 0
    total_duration = 0
    for file_path in sample_files:
        try:
            data = np.load(file_path)
            fps = data['fps'].item()
            frames = data['body_positions'].shape[0]
            duration = frames / fps
            total_frames += frames
            total_duration += duration
            
            print(f"   📄 {file_path.name}: {frames}帧, {duration:.1f}秒, {fps}fps")
        except Exception as e:
            print(f"   ❌ {file_path.name}: 读取失败 - {e}")
    
    if sample_files:
        avg_frames = total_frames / len(sample_files)
        avg_duration = total_duration / len(sample_files)
        print(f"\n📊 样本平均: {avg_frames:.0f}帧, {avg_duration:.1f}秒")

def list_dataset_contents(dataset_name: str, max_subjects: int = 10):
    """列出数据集内容"""
    dataset_path = Path("g1") / dataset_name
    if not dataset_path.exists():
        print(f"❌ 数据集 '{dataset_name}' 不存在")
        return
    
    subjects = [s for s in dataset_path.iterdir() if s.is_dir()]
    subjects.sort()
    
    print(f"📂 数据集: {dataset_name}")
    print(f"📁 子目录数量: {len(subjects)}")
    
    for i, subject in enumerate(subjects[:max_subjects]):
        files = list(subject.glob("*.npz"))
        print(f"\n   📂 {subject.name} ({len(files)} 个文件):")
        
        # 显示前几个文件
        for file_path in files[:3]:
            try:
                data = np.load(file_path)
                frames = data['body_positions'].shape[0]
                fps = data['fps'].item()
                duration = frames / fps
                print(f"      📄 {file_path.name}: {frames}帧, {duration:.1f}秒")
            except:
                print(f"      📄 {file_path.name}: 读取失败")
        
        if len(files) > 3:
            print(f"      ... 还有 {len(files) - 3} 个文件")
    
    if len(subjects) > max_subjects:
        print(f"\n   ... 还有 {len(subjects) - max_subjects} 个子目录")

def preview_motion(file_path: str, max_frames: int = 10):
    """预览动作数据"""
    try:
        data = np.load(file_path)
        fps = data['fps'].item()
        body_positions = data['body_positions']
        body_rotations = data['body_rotations']
        dof_positions = data['dof_positions']
        
        print(f"🎬 动作预览: {file_path}")
        print(f"📊 基本信息:")
        print(f"   - 帧数: {body_positions.shape[0]}")
        print(f"   - FPS: {fps}")
        print(f"   - 时长: {body_positions.shape[0]/fps:.2f}秒")
        print(f"   - 身体部位: {body_positions.shape[1]}")
        print(f"   - 关节数: {dof_positions.shape[1]}")
        
        print(f"\n📈 数据预览 (前{max_frames}帧):")
        print(f"   帧  根位置(X,Y,Z)          关节角度范围")
        print(f"   --- ---------------------- ------------")
        
        for i in range(min(max_frames, body_positions.shape[0])):
            root_pos = body_positions[i, 0, :]
            joint_range = f"[{dof_positions[i].min():.2f}, {dof_positions[i].max():.2f}]"
            print(f"   {i:3d} [{root_pos[0]:6.3f}, {root_pos[1]:6.3f}, {root_pos[2]:6.3f}] {joint_range}")
        
        # 统计信息
        print(f"\n📊 统计信息:")
        print(f"   位置范围: X[{body_positions[:,:,0].min():.3f}, {body_positions[:,:,0].max():.3f}]")
        print(f"             Y[{body_positions[:,:,1].min():.3f}, {body_positions[:,:,1].max():.3f}]")
        print(f"             Z[{body_positions[:,:,2].min():.3f}, {body_positions[:,:,2].max():.3f}]")
        print(f"   关节角度: [{dof_positions.min():.3f}, {dof_positions.max():.3f}]")
        
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")

def find_short_motions(dataset_name: str, max_duration: float = 5.0):
    """查找短动作（适合测试）"""
    dataset_path = Path("g1") / dataset_name
    if not dataset_path.exists():
        print(f"❌ 数据集 '{dataset_name}' 不存在")
        return
    
    short_motions = []
    
    for subject in dataset_path.iterdir():
        if subject.is_dir():
            for file_path in subject.glob("*.npz"):
                try:
                    data = np.load(file_path)
                    fps = data['fps'].item()
                    frames = data['body_positions'].shape[0]
                    duration = frames / fps
                    
                    if duration <= max_duration:
                        short_motions.append({
                            'file': file_path,
                            'duration': duration,
                            'frames': frames,
                            'subject': subject.name
                        })
                except:
                    continue
    
    if short_motions:
        print(f"🔍 找到 {len(short_motions)} 个时长 ≤ {max_duration}秒 的动作:")
        print(f"{'子目录':<12} {'文件名':<25} {'时长(s)':<8} {'帧数':<6}")
        print("-" * 55)
        
        for motion in sorted(short_motions, key=lambda x: x['duration']):
            print(f"{motion['subject']:<12} {motion['file'].name:<25} "
                  f"{motion['duration']:<8.2f} {motion['frames']:<6}")
    else:
        print(f"❌ 未找到时长 ≤ {max_duration}秒 的动作")

def main():
    parser = argparse.ArgumentParser(description="AMASS数据集快速查看工具")
    parser.add_argument("--action", choices=["stats", "list", "preview", "short"], 
                       default="stats", help="执行的操作")
    parser.add_argument("--dataset", help="数据集名称")
    parser.add_argument("--file", help="文件路径")
    parser.add_argument("--max_duration", type=float, default=5.0, help="最大时长（秒）")
    
    args = parser.parse_args()
    
    if args.action == "stats":
        quick_stats()
    elif args.action == "list":
        if not args.dataset:
            print("❌ 需要指定 --dataset 参数")
            return
        list_dataset_contents(args.dataset)
    elif args.action == "preview":
        if not args.file:
            print("❌ 需要指定 --file 参数")
            return
        preview_motion(args.file)
    elif args.action == "short":
        if not args.dataset:
            print("❌ 需要指定 --dataset 参数")
            return
        find_short_motions(args.dataset, args.max_duration)

if __name__ == "__main__":
    main() 