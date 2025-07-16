#!/usr/bin/env python3
"""
修复版本的MuJoCo动作播放器
解决了关节数量不匹配的问题
"""

import numpy as np
import time
import sys
import os
import argparse
import mujoco
import mujoco.viewer
from pathlib import Path
from enum import Enum
from typing import Optional, Union, Dict, Any, Tuple
import re
from tqdm import tqdm

class FileType(Enum):
    """Enum for different motion file types"""
    STAGE0 = "stage0"  # PoseLib .npy format
    STAGE1 = "stage1"  # PoseLib .npy format
    STAGE2 = "stage2"  # Final .npz format
    AUTO = "auto"      # Auto-detect based on extension

class MotionVisualizer:
    """
    A class to visualize motion files in MuJoCo.
    
    Supports different file formats and provides options for playback control.
    """
    # Constants
    DEFAULT_MODEL_PATH = 'g1_description/g1_29dof_rev_1_0.xml'  # 使用29DOF模型
    JOINT_NAMES = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
        'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
        'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
    ]
    
    def __init__(self, model_path: str = None):
        """
        Initialize the visualizer.
        
        Args:
            model_path: Path to the MuJoCo XML model file. If None, uses default path.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self._verify_model_exists()
        
        # Initialize motion data
        self.motion_data = None
        self.fps = None
        self.joint_positions = None
        self.num_frames = 0
        self.file_path = None
        self.file_type = None
        
        # Add running flag for playback control
        self.running = False
    
    def _verify_model_exists(self):
        """Verify that the MuJoCo model file exists"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo model file not found: {self.model_path}")
    
    def load_motion(self, file_path: str, file_type: FileType = FileType.AUTO) -> bool:
        """
        Load a motion file.
        
        Args:
            file_path: Path to the motion file
            file_type: Type of file to load (auto-detect by default)
            
        Returns:
            True if loading was successful, False otherwise
        """
        self.file_path = file_path
        
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        
        # Auto-detect file type if not specified
        if file_type == FileType.AUTO:
            file_type = self._detect_file_type(file_path)
        
        self.file_type = file_type
        
        try:
            if file_type == FileType.STAGE1:
                return self._load_stage1()
            elif file_type == FileType.STAGE2:
                return self._load_stage2()
            elif file_type == FileType.STAGE0:
                return self._load_stage0()
            else:
                print(f"Error: Unsupported file type: {file_type}")
                return False
        except Exception as e:
            print(f"Error loading motion file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_file_type(self, file_path: str) -> FileType:
        """Detect file type based on extension and content"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npz':
            return FileType.STAGE2
        elif ext == '.npy':
            if '_intermediate' in file_path:
                return FileType.STAGE1
            else:
                return FileType.STAGE0
                
        # Default to Stage2 if we can't determine
        print(f"Warning: Could not auto-detect file type for {file_path}. Assuming Stage2.")
        return FileType.STAGE2

    def _load_stage0(self) -> bool:
        """Load a stage0 file (Vanilla AMASS .npy format)"""
        try:
            self.joint_positions = np.load(self.file_path)
            self.fps = int(re.sub(r'\D', '', self.file_path[-12:-9]))

            # convert to motion data format expected by MuJoCo
            self.num_frames = self.joint_positions.shape[0]
            self.joint_positions[:, 2] += 0.793

            print(f"Loaded Stage0 file with {self.num_frames} frames at {self.fps} fps")
            return True
        except Exception as e:
            print(f"Error loading Stage0 file: {e}")
            return False
    
    def _load_stage1(self) -> bool:
        """Load a Stage1 file (PoseLib .npy format)"""
        try:
            # Only import specialized modules when needed
            try:
                from poselib.skeleton.skeleton3d import SkeletonMotion
                from isaac_utils.rotations import get_euler_xyz
            except ImportError:
                print("Error: PoseLib modules required for Stage1 file support")
                return False
            
            motion = SkeletonMotion.from_file(self.file_path)
            self.fps = motion.fps
            
            # Convert the motion data to the format expected by MuJoCo
            local_rotations = motion.local_rotation
            root_translation = motion.root_translation
            
            self.num_frames = local_rotations.shape[0]
            self.joint_positions = np.zeros((self.num_frames, 36))
            
            # Set root position and rotation
            self.joint_positions[:, :3] = root_translation
            self.joint_positions[:, 3:7] = local_rotations[:, 0, :]
            
            # Convert joint rotations to angles
            self.joint_positions[:, 7:] = self._to_angle(local_rotations[:, 1:, :])
            
            print(f"Loaded Stage1 file with {self.num_frames} frames at {self.fps} fps")
            return True
            
        except Exception as e:
            print(f"Error loading Stage1 file: {e}")
            return False

    def _load_stage2(self) -> bool:
        """Load a Stage2 file (final .npz format)"""
        try:
            data = np.load(self.file_path)
            self.fps = data['fps'].item()
            
            # Extract the data
            body_positions = data['body_positions']
            body_rotations = data['body_rotations']
            dof_positions = data['dof_positions']
            
            self.num_frames = body_positions.shape[0]
            num_joints = dof_positions.shape[1]
            
            # 创建正确大小的数组 (7 + num_joints)
            # 7 = 3 (root position) + 4 (root rotation)
            self.joint_positions = np.zeros((self.num_frames, 7 + num_joints))
            
            # Set root position and rotation
            self.joint_positions[:, :3] = body_positions[:, 0, :]
            self.joint_positions[:, 3:7] = body_rotations[:, 0, :]
            
            # Set joint angles
            self.joint_positions[:, 7:] = dof_positions
            
            print(f"Loaded Stage2 file with {self.num_frames} frames at {self.fps} fps")
            print(f"Data shape: {self.joint_positions.shape} (7 + {num_joints} joints)")
            return True
            
        except Exception as e:
            print(f"Error loading Stage2 file: {e}")
            return False
    
    def _to_angle(self, data_rot):
        """Convert quaternions to Euler angles based on joint types"""
        try:
            import torch
            from isaac_utils.rotations import get_euler_xyz
        except ImportError:
            print("Error: PoseLib modules required for Stage1 file support")
            
        num_frames, num_joints, _ = data_rot.shape
        joint_angles = torch.zeros((num_frames, num_joints), device=data_rot.device)
        
        # Determine primary axis for each joint
        joint_axes = []
        for name in self.JOINT_NAMES:
            if 'roll' in name:
                joint_axes.append(0)  # x-axis (roll)
            elif any(term in name for term in ['pitch', 'knee', 'elbow']):
                joint_axes.append(1)  # y-axis (pitch)
            else:
                joint_axes.append(2)  # z-axis (yaw)

        for i in range(num_joints):
            # Get quaternions for this joint across all frames
            joint_quat = data_rot[:, i, :]
            
            # Convert to Euler angles
            roll, pitch, yaw = get_euler_xyz(joint_quat, w_last=True)
            
            # Select the appropriate angle based on the joint's primary axis
            if joint_axes[i] == 0:  # x-axis (roll)
                joint_angles[:, i] = roll
            elif joint_axes[i] == 1:  # y-axis (pitch)
                joint_angles[:, i] = pitch
            else:  # z-axis (yaw)
                joint_angles[:, i] = yaw

        return joint_angles.cpu().numpy()
        
    def stop(self) -> None:
        """Stop the current playback"""
        self.running = False

    def play(self, 
         loop: bool = True, 
         start_frame: int = 0, 
         end_frame: Optional[int] = None,
         record_video: bool = False,
         video_path: Optional[str] = None,
         camera_distance: float = 8.0,
         speed_factor: float = 1.0,
         incidents: Optional[Dict] = None) -> None:
    
        if self.joint_positions is None:
            print("Error: No motion data loaded. Call load_motion() first.")
            return
        
        # Set running flag to True at start of playback
        self.running = True
        
        # ANSI color codes
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m"
        }
        
        # Initialize MuJoCo model and data
        model = mujoco.MjModel.from_xml_path(self.model_path)
        model.opt.timestep = 1.0 / (self.fps * speed_factor)  # Apply speed factor
        data = mujoco.MjData(model)
        
        # 检查数据维度匹配
        data_dim = self.joint_positions.shape[1]
        model_dim = data.qpos.shape[0]
        
        print(f"Data dimension: {data_dim}, Model dimension: {model_dim}")
        
        if data_dim != model_dim:
            print(f"Warning: Data dimension ({data_dim}) doesn't match model dimension ({model_dim})")
            print("Attempting to pad or truncate data...")
            
            # 如果数据维度小于模型维度，用零填充
            if data_dim < model_dim:
                padded_data = np.zeros((self.num_frames, model_dim))
                padded_data[:, :data_dim] = self.joint_positions
                self.joint_positions = padded_data
                print(f"Padded data to match model dimension: {model_dim}")
            else:
                # 如果数据维度大于模型维度，截断数据
                self.joint_positions = self.joint_positions[:, :model_dim]
                print(f"Truncated data to match model dimension: {model_dim}")
        
        # Set frame range
        if end_frame is None:
            end_frame = self.num_frames
        else:
            end_frame = min(end_frame, self.num_frames)
            
        if start_frame >= end_frame:
            print(f"Error: Invalid frame range ({start_frame}-{end_frame})")
            return
            
        # Calculate total duration
        frame_count = end_frame - start_frame
        total_duration = frame_count / self.fps
        
        print(f"Playing frames {start_frame}-{end_frame} ({total_duration:.2f}s)")
        
        # If incidents are provided, print them as a legend
        if incidents:
            print("\nMarked incidents:")
            for name, info in incidents.items():
                start = info.get("start_time", 0)
                end = info.get("end_time", 0)
                color = info.get("color", "red")
                color_code = colors.get(color, colors["red"])
                print(f"  {color_code}■{colors['reset']} {name}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
            print("")
        
        # Generate video path if needed
        if record_video and video_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            video_path = f"{base_name}_recording.mp4"
        
        # Function to create a color-coded timeline bar
        def create_timeline(current_time, total_time, width=50):
            # Calculate the position of the current time in the bar
            bar_width = width
            position = int((current_time / total_time) * bar_width) if total_time > 0 else 0
            
            # Create the base bar
            bar = ['█'] * bar_width
            
            # Apply colors to segments representing incidents
            if incidents:
                for name, info in incidents.items():
                    start = info.get("start_time", 0)
                    end = info.get("end_time", 0)
                    color = info.get("color", "red")
                    color_code = colors.get(color, colors["red"])
                    
                    # Convert time to bar positions
                    start_pos = max(0, min(bar_width-1, int((start / total_time) * bar_width)))
                    end_pos = max(0, min(bar_width-1, int((end / total_time) * bar_width)))
                    
                    # Apply color to this segment
                    for i in range(start_pos, end_pos + 1):
                        if i < len(bar):
                            bar[i] = f"{color_code}█{colors['reset']}"
            
            # Create the final bar string
            timeline = ''.join(bar[:position]) + '⚫' + ''.join(bar[position+1:]) if position < bar_width else ''.join(bar)
            return timeline
        
        # Start the viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            # Configure camera
            viewer.cam.distance = camera_distance
            
            # Start recording if requested
            if record_video:
                print(f"Recording to {video_path}")
                viewer.record_start(video_path, fps=int(self.fps))  # Ensure FPS is integer
            
            step = start_frame
            done = False
            last_update_time = 0
            
            # Hide cursor during playback for cleaner display
            print("\033[?25l", end="", flush=True)  # Hide cursor
            
            try:
                # Main playback loop
                while viewer.is_running() and not done and self.running:
                    step_start = time.time()
                    
                    # Apply joint positions to MuJoCo model
                    data.qpos = self.joint_positions[step, :]
                    
                    # Quaternion needs special handling based on file type
                    if self.file_type == FileType.STAGE0 or self.file_type == FileType.STAGE1:
                        # Reorder quaternion to [w, x, y, z] for stage1 files
                        if data.qpos.shape[0] >= 7:
                            data.qpos[3] = self.joint_positions[step, 6]   # w
                            data.qpos[4:7] = self.joint_positions[step, 3:6]  # x,y,z
                    else:
                        # Stage2 files already have the correct quaternion order
                        if data.qpos.shape[0] >= 7:
                            data.qpos[3:7] = self.joint_positions[step, 3:7]
                    
                    # Update MuJoCo
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    
                    # Update progress bar (only update every 0.1 seconds to avoid flicker)
                    current_time = (step - start_frame) / self.fps
                    if current_time - last_update_time >= 0.1:
                        # Generate colored timeline
                        timeline = create_timeline(current_time, total_duration)
                        
                        # Print the progress with custom colored timeline
                        elapsed = time.time() - step_start
                        remaining = (total_duration - current_time) / speed_factor if current_time < total_duration else 0
                        
                        # Clear the current line and print the new status
                        print(f"\rPlayback: {timeline} {current_time:.1f}s/{total_duration:.1f}s", end="", flush=True)
                        
                        last_update_time = current_time
                    
                    # Update frame counter
                    step += 1
                    
                    # Handle end of sequence
                    if step >= end_frame:
                        if loop:
                            step = start_frame  # Loop back to start
                            last_update_time = 0
                        else:
                            print(f"\rPlayback: {create_timeline(total_duration, total_duration)} {total_duration:.1f}s/{total_duration:.1f}s", flush=True)
                            print()  # New line
                            
                            if record_video:
                                print(f"Video saved to {video_path}")
                                viewer.record_stop()
                            done = True  # Exit if not looping
                    
                    # Control playback speed
                    dt = model.opt.timestep - (time.time() - step_start)
                    if dt > 0:
                        time.sleep(dt)
                        
            finally:
                # Always restore cursor visibility when done
                print("\033[?25h", end="", flush=True)  # Show cursor
                print()  # New line
                
                # Ensure recording is stopped when viewer is closed
                if record_video and viewer.is_recording():
                    viewer.record_stop()
                    print(f"Video saved to {video_path}")

def parse_args():
    """Parse command-line arguments for the visualizer"""
    parser = argparse.ArgumentParser(description="Visualize motion files in MuJoCo (Fixed Version)")
    
    parser.add_argument('file_path', type=str, help='Path to the motion file')
    
    parser.add_argument('--file-type', type=str, choices=['auto', 'stage1', 'stage2'], 
                        default='auto', help='Type of motion file')
    
    parser.add_argument('--model', type=str, 
                        default="/home/xunyang/Desktop/Projects/Genesis_Legged_Gym/resources/robots/g1_description/g1_29dof_rev_1_0.xml",
                        help='Path to MuJoCo XML model file')
    
    parser.add_argument('--no-loop', action='store_true',
                        help='Play once instead of looping')
    
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Frame to start playback from')
    
    parser.add_argument('--end-frame', type=int, default=None,
                        help='Frame to end playback')
    
    parser.add_argument('--record', action='store_true',
                        help='Record playback as video')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save video file')
    
    parser.add_argument('--camera-distance', type=float, default=8.0,
                        help='Camera distance from subject')
    
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier')
    
    return parser.parse_args()

def main():
    """Main function for command-line usage"""
    args = parse_args()
    
    # Map file type string to enum
    file_type_map = {
        'auto': FileType.AUTO,
        'stage1': FileType.STAGE1,
        'stage2': FileType.STAGE2
    }
    
    # Initialize visualizer
    visualizer = MotionVisualizer(model_path=args.model)
    
    # Load motion file
    if not visualizer.load_motion(args.file_path, file_type=file_type_map[args.file_type]):
        sys.exit(1)
    
    # Play or record the motion
    visualizer.play(
        loop=not args.no_loop,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        record_video=args.record,
        video_path=args.output,
        camera_distance=args.camera_distance,
        speed_factor=args.speed
    )

if __name__ == "__main__":
    main() 