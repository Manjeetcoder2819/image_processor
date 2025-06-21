import cv2
import numpy as np
import tempfile
import os
from advanced_detector import AdvancedShapeDetector

class VideoShapeProcessor:
    def __init__(self, min_area=100, epsilon_factor=0.02):
        """
        Video processor for extracting shapes from video files
        """
        self.detector = AdvancedShapeDetector(min_area, epsilon_factor)
        
    def process_video_file(self, video_file, sample_rate=30):
        """
        Process uploaded video file and extract shapes from frames
        
        Args:
            video_file: Streamlit uploaded video file
            sample_rate: Process every Nth frame
            
        Returns:
            Dictionary with processed frames and detected shapes
        """
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_video_path = tmp_file.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(tmp_video_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_info = {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps > 0 else 0
            }
            
            processed_frames = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every sample_rate frames
                if frame_number % sample_rate == 0:
                    # Detect shapes in this frame
                    shapes = self.detector.detect_shapes_advanced(frame)
                    
                    if shapes:  # Only keep frames with detected shapes
                        # Draw shapes on frame
                        processed_frame = self.detector.draw_advanced_contours(frame.copy(), shapes)
                        
                        processed_frames.append({
                            'frame_number': frame_number,
                            'timestamp': frame_number / fps if fps > 0 else 0,
                            'original_frame': frame,
                            'processed_frame': processed_frame,
                            'shapes': shapes,
                            'shape_count': len(shapes)
                        })
                
                frame_number += 1
                
                # Limit to prevent memory issues
                if len(processed_frames) >= 50:
                    break
            
            cap.release()
            
            return {
                'video_info': video_info,
                'processed_frames': processed_frames,
                'total_processed': len(processed_frames)
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
    
    def extract_key_frames(self, video_result):
        """
        Extract key frames with the most interesting shapes
        
        Args:
            video_result: Result from process_video_file
            
        Returns:
            List of key frames with most diverse shapes
        """
        if not video_result['processed_frames']:
            return []
        
        # Score frames based on shape diversity and count
        scored_frames = []
        
        for frame_data in video_result['processed_frames']:
            shapes = frame_data['shapes']
            
            # Calculate diversity score
            shape_types = set(shape['type'] for shape in shapes)
            diversity_score = len(shape_types)
            
            # Calculate total area of shapes
            total_area = sum(shape['area'] for shape in shapes)
            
            # Calculate complexity score
            complexity_score = sum(
                shape.get('complexity_score', 0.5) for shape in shapes
            ) / len(shapes) if shapes else 0
            
            # Combined score
            score = diversity_score * 2 + len(shapes) + (total_area / 10000) + complexity_score
            
            scored_frames.append({
                'frame_data': frame_data,
                'score': score,
                'diversity': diversity_score,
                'total_area': total_area,
                'complexity': complexity_score
            })
        
        # Sort by score and return top frames
        scored_frames.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top 10 key frames
        return [item['frame_data'] for item in scored_frames[:10]]
    
    def create_shape_timeline(self, video_result):
        """
        Create a timeline of shape detection throughout the video
        
        Args:
            video_result: Result from process_video_file
            
        Returns:
            Timeline data for visualization
        """
        timeline = []
        
        for frame_data in video_result['processed_frames']:
            timeline_entry = {
                'timestamp': frame_data['timestamp'],
                'frame_number': frame_data['frame_number'],
                'shape_count': frame_data['shape_count'],
                'shape_types': {}
            }
            
            # Count shape types in this frame
            for shape in frame_data['shapes']:
                shape_type = shape['type']
                if shape_type not in timeline_entry['shape_types']:
                    timeline_entry['shape_types'][shape_type] = 0
                timeline_entry['shape_types'][shape_type] += 1
            
            timeline.append(timeline_entry)
        
        return timeline
    
    def generate_video_summary_report(self, video_result):
        """
        Generate a comprehensive report of shapes detected in video
        
        Args:
            video_result: Result from process_video_file
            
        Returns:
            Summary report dictionary
        """
        if not video_result['processed_frames']:
            return {'error': 'No frames with shapes detected'}
        
        all_shapes = []
        for frame_data in video_result['processed_frames']:
            all_shapes.extend(frame_data['shapes'])
        
        # Shape type distribution
        shape_distribution = {}
        total_area = 0
        
        for shape in all_shapes:
            shape_type = shape['type']
            if shape_type not in shape_distribution:
                shape_distribution[shape_type] = {
                    'count': 0,
                    'total_area': 0,
                    'avg_area': 0,
                    'max_area': 0,
                    'min_area': float('inf')
                }
            
            shape_distribution[shape_type]['count'] += 1
            shape_distribution[shape_type]['total_area'] += shape['area']
            shape_distribution[shape_type]['max_area'] = max(
                shape_distribution[shape_type]['max_area'], shape['area']
            )
            shape_distribution[shape_type]['min_area'] = min(
                shape_distribution[shape_type]['min_area'], shape['area']
            )
            total_area += shape['area']
        
        # Calculate averages
        for shape_type in shape_distribution:
            count = shape_distribution[shape_type]['count']
            shape_distribution[shape_type]['avg_area'] = (
                shape_distribution[shape_type]['total_area'] / count
            )
        
        # Video statistics
        video_stats = {
            'total_shapes_detected': len(all_shapes),
            'unique_shape_types': len(shape_distribution),
            'frames_with_shapes': len(video_result['processed_frames']),
            'total_frames_processed': video_result['video_info']['total_frames'],
            'detection_rate': len(video_result['processed_frames']) / max(1, video_result['video_info']['total_frames']),
            'avg_shapes_per_frame': len(all_shapes) / max(1, len(video_result['processed_frames'])),
            'total_shape_area': total_area
        }
        
        return {
            'video_info': video_result['video_info'],
            'video_stats': video_stats,
            'shape_distribution': shape_distribution,
            'timeline': self.create_shape_timeline(video_result),
            'key_frames': self.extract_key_frames(video_result)
        }