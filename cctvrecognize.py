import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from test import MultimodalBiometricSystem

class CCTVRecognizer:
    def __init__(self, system, target_fps: int = 10):
        """
        Initialize the CCTV recognizer
        
        Args:
            system: Initialized MultimodalBiometricSystem
            target_fps: Desired processing frames per second
        """
        self.system = system
        self.target_fps = target_fps
        self.frame_count = 0
        self.tracked_persons = {}
        self.next_track_id = 1
        self.last_frame_time = 0

    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process video file with recognition
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame processing interval
        frame_interval = max(1, int(round(input_fps / self.target_fps)))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.target_fps, (width, height))

        print(f"Processing: {video_path}")
        print(f"Original: {input_fps:.1f} FPS, {total_frames} frames")
        print(f"Processing at: {self.target_fps} FPS (1 every {frame_interval} frames)")
        
        start_time = time.time()
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            
            # Skip frames to maintain target FPS
            if self.frame_count % frame_interval != 0:
                continue

            # Process frame with timing control
            current_time = time.time()
            if current_time - self.last_frame_time < 1/self.target_fps:
                continue
            self.last_frame_time = current_time

            # Perform recognition
            processed_frame, results = self.process_frame(frame)
            processed_frames += 1
            
            # Display processing progress
            progress = (self.frame_count / total_frames) * 100
            cv2.putText(processed_frame, f"Progress: {progress:.1f}%", 
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write or display frame
            if writer:
                writer.write(processed_frame)
            else:
                cv2.imshow("CCTV Recognition", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        processing_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"Frames processed: {processed_frames}/{total_frames}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {processed_frames/processing_time:.2f}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for recognition
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (processed_frame, recognition_results)
        """
        # Perform recognition
        processed_frame, results = self.system.recognize_person(frame)
        
        # Update tracking
        self.update_tracking(results)
        
        # Draw tracking information
        for person_id, info in self.tracked_persons.items():
            if info['last_seen'] == self.frame_count:
                x1, y1, x2, y2 = info['bbox']
                color = (0, 255, 0) if info['name'] != 'Unknown' else (0, 0, 255)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, f"ID:{person_id} {info['name']}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display frame info
        cv2.putText(processed_frame, f"Frame: {self.frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Tracking: {len(self.tracked_persons)} persons", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return processed_frame, results

    def update_tracking(self, results: List[Dict]):
        """
        Maintain tracking of recognized persons across frames
        
        Args:
            results: List of recognition results from current frame
        """
        # Remove stale tracks (not seen for 30 frames)
        to_delete = [pid for pid, info in self.tracked_persons.items() 
                    if self.frame_count - info['last_seen'] > 30]
        for pid in to_delete:
            del self.tracked_persons[pid]
        
        # Update existing tracks or create new ones
        for result in results:
            # Skip unknown persons
            if result['name'] == 'Unknown':
                continue
                
            # Try to match with existing tracks
            matched_id = None
            for pid, info in self.tracked_persons.items():
                if info['name'] == result['name']:
                    matched_id = pid
                    break
                    
            if matched_id:
                # Update existing track
                self.tracked_persons[matched_id].update({
                    'last_seen': self.frame_count,
                    'bbox': result.get('bbox', (0, 0, 0, 0))
                })
            else:
                # Create new track
                self.tracked_persons[self.next_track_id] = {
                    'name': result['name'],
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'bbox': result.get('bbox', (0, 0, 0, 0))
                }
                self.next_track_id += 1


if __name__ == "__main__":
    # Initialize your biometric system (make sure to import or define it)
    # Adjust import as needed
    
    # Create system instance
    system = MultimodalBiometricSystem("best_3.pt")  # Your ear model path
    
    # Initialize recognizer with desired processing speed (5-15 FPS recommended)
    recognizer = CCTVRecognizer(system, target_fps=20)
    
    # Process video
    input_video = "1.mp4"  # Your input video path
    output_video = "output_recognition.mp4"  # Output path (optional)
    
    # Start processing
    recognizer.process_video(input_video, output_video)