import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class MultimodalBiometricSystem:
    def __init__(self, ear_model_path: str):
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        
        # Initialize ArcFace R100
        self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))

        # Load YOLO model for detecting Human Ears
        try:
            self.ear_detector = YOLO(ear_model_path) 
            self.ear_detector.fuse()  # Fuse model layers for optimization
            self.ear_detector.to(self.device)
            print("Ear detector model loaded successfully!")
        except Exception as e:
            print(f"Error loading ear detector model: {e}")
        self.ear_encoder = self._init_ear_encoder()
        
        # Database
        self.known_face_embeddings = []
        self.known_ear_embeddings = []
        self.known_names = []
        self.load_database()
        
        # Thresholds
        self.FACE_THRESHOLD = 0.35  # ArcFace generally needs lower threshold
        self.EAR_THRESHOLD = 0.50
        self.COMBINED_CONFIDENCE = 0.98

    def _init_ear_encoder(self):
        """Initialize ResNet50 for ear feature extraction"""
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        return {'model': model, 'preprocess': preprocess}

    def load_database(self):
        """Load registered biometrics from database"""
        if not os.path.exists("biometric_database"):
            os.makedirs("biometric_database")
            return
            
        self.known_face_embeddings = []
        self.known_ear_embeddings = []
        self.known_names = []
        
        for filename in os.listdir("biometric_database"):
            if filename.endswith('.pkl'):
                try:
                    with open(os.path.join("biometric_database", filename), 'rb') as f:
                        data = pickle.load(f)
                        if 'face_embedding' in data and 'ear_embedding' in data and 'name' in data:
                            self.known_face_embeddings.append(data['face_embedding'])
                            self.known_ear_embeddings.append(data['ear_embedding'])
                            self.known_names.append(data['name'])
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    def register_person(self, name: str) -> bool:
        """Register a new person with face and ear samples"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False

        face_samples = []
        ear_samples = []
        sample_count = 0
        required_samples = 5  # One sample for each direction: frontal, left, right, up, down

        angles = ['frontal', 'left', 'right', 'up', 'down']
        
        print(f"Registering {name}. Please move your head to capture different angles.")

        for angle in angles:
            while cap.isOpened() and sample_count < len(angles):
                ret, frame = cap.read()
                if not ret:
                    continue

                # Display instructions
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Sample {sample_count+1}/{len(angles)}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Move your head to {angle}", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                # Process face and ears
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.arcface.get(rgb_frame)
                
                # Draw face detections
                if faces:
                    face = faces[0]  # Take the most prominent face
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Process ears
                ear_results = self.ear_detector(rgb_frame, verbose=False)
                for box in ear_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.imshow(f"Registering {name}", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                elif key == ord('c'):
                    if faces:
                        face_samples.append(faces[0].embedding)
                        sample_count += 1
                        print(f"Captured {angle} sample {sample_count}/{len(angles)}")
                        break  # Move to next angle after capture

        cap.release()
        cv2.destroyAllWindows()

        if len(face_samples) >= 3:  # Require at least 3 good samples
            avg_face = np.mean(face_samples, axis=0)
            
            # Get ear samples from the last frame (if any)
        if ear_results:
            print(f"Ear detections found: {len(ear_results[0].boxes)}")
            for box in ear_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Ear box coordinates: {x1},{y1} - {x2},{y2}")
                
                # Ensure coordinates are valid
                if x2 <= x1 or y2 <= y1:
                    print("Invalid ear box coordinates!")
                    continue
                    
                ear_img = rgb_frame[y1:y2, x1:x2]
                print(f"Ear image shape: {ear_img.shape}")
                
                if ear_img.size > 0 and ear_img.shape[0] > 10 and ear_img.shape[1] > 10:
                    ear_embedding = self.encode_ear(ear_img)
                    print(f"Encoding successful: {ear_embedding is not None}")
                    
                    if ear_embedding is not None:
                        ear_samples.append(ear_embedding)
                        print(f"Ear sample added. Total samples: {len(ear_samples)}")
                        print(f"Sample shape: {ear_embedding.shape}")
                else:
                    print("Empty or too small ear image")
        else:
            print("No ear detections in this frame")
            
            avg_ear = np.mean(ear_samples, axis=0) if ear_samples else np.zeros(2048)  # Default if no ears

            data = {
                'name': name,
                'face_embedding': avg_face,
                'ear_embedding': avg_ear,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

            filename = f"biometric_database/{name}_{data['timestamp']}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(data, f)

            self.load_database()  # Refresh database
            print(f"Successfully registered {name}!")
            return True
        # else:
        #     print(f"Registration failed. Got {len(face_samples)} face samples")
        #     return False

    def recognize_person(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Recognize a person in the given frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = frame.copy()
        results = []
        
        # Process face
        face_embeddings, face_boxes = self._process_face(rgb_frame, return_box=True)
        
        # Process ears
        ear_embeddings, ear_boxes = self._process_ears(rgb_frame, return_boxes=True)
        
        # Multimodal matching
        if face_embeddings:
            for i, face_embedding in enumerate(face_embeddings):
                match_result = self._multimodal_match(face_embedding, ear_embeddings)
                if match_result:
                    results.append(match_result)
                    # Draw face box with recognition result
                    if i < len(face_boxes):
                        x1, y1, x2, y2 = face_boxes[i]
                        color = (0, 255, 0) if match_result['name'] != 'Unknown' else (0, 0, 255)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        text = f"{match_result['name']} ({match_result['confidence']*100:.1f}%)"
                        cv2.putText(output_frame, text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw ear boxes
        for box in ear_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return output_frame, results

    def _process_face(self, rgb_frame: np.ndarray, return_box: bool = False):
        """Use ArcFace for face detection and embedding"""
        try:
            faces = self.arcface.get(rgb_frame)
            if not faces:
                return (None, None) if return_box else None
            
            embeddings = [face.embedding for face in faces]
            boxes = [face.bbox.astype(int) for face in faces]
            
            return (embeddings, boxes) if return_box else embeddings
        except Exception as e:
            print(f"Face processing error: {e}")
            return (None, None) if return_box else None

    def _process_ears(self, rgb_frame: np.ndarray, return_boxes: bool = False):
        """Detect ears and extract features"""
        try:
            ear_results = self.ear_detector(rgb_frame, verbose=False)
            embeddings = []
            boxes = []
            
            for box in ear_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ear_img = rgb_frame[y1:y2, x1:x2]
                
                if ear_img.size == 0:
                    continue
                    
                embedding = self.encode_ear(ear_img)
                if embedding is not None:
                    embeddings.append(embedding)
                    boxes.append((x1, y1, x2, y2))
            
            return (embeddings, boxes) if return_boxes else embeddings
        except Exception as e:
            print(f"Ear processing error: {e}")
            return ([], []) if return_boxes else []

    def encode_ear(self, ear_img: np.ndarray) -> Optional[np.ndarray]:
        """Convert ear image to feature vector"""
        try:
            input_tensor = self.ear_encoder['preprocess'](ear_img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.ear_encoder['model'](input_batch)
            return features.squeeze().cpu().numpy().flatten()
        except Exception as e:
            print(f"Ear encoding error: {e}")
            return None

    def _multimodal_match(self, face_embedding: np.ndarray, 
                         ear_embeddings: List[np.ndarray]) -> Dict:
        """Fuse face and ear recognition results"""
        face_match = self._match_face(face_embedding)
        ear_match = self._match_ear(ear_embeddings) if ear_embeddings else {'name': 'Unknown', 'similarity': 0.0}
        
        # Case 1: Both modalities agree
        if (face_match['name'] != 'Unknown' and ear_match['name'] != 'Unknown' and 
            face_match['name'] == ear_match['name']):
            combined_conf = min(1.0, (face_match['similarity'] + ear_match['similarity']) / 2)
            return {
                'name': face_match['name'],
                'confidence': combined_conf,
                'modality': 'face+ear',
                'face_similarity': face_match['similarity'],
                'ear_similarity': ear_match['similarity']
            }
    
        
        # Case 2: Only face matches
        elif face_match['name'] != 'Unknown':
            return {
                'name': face_match['name'],
                'confidence': face_match['similarity'],
                'modality': 'face',
                'face_similarity': face_match['similarity'],
                'ear_similarity': None
            }
        
        # Case 3: Only ear matches
        elif ear_match['name'] != 'Unknown':
            return {
                'name': ear_match['name'],
                'confidence': ear_match['similarity'],
                'modality': 'ear',
                'face_similarity': None,
                'ear_similarity': ear_match['similarity']
            }
        
        # Case 4: No matches
        return {
            'name': 'Unknown',
            'confidence': 0.0,
            'modality': 'none',
            'face_similarity': face_match['similarity'],
            'ear_similarity': ear_match['similarity']
        }

    def _match_face(self, embedding: np.ndarray) -> Dict:
        """Match face against database"""
        if not self.known_face_embeddings:
            return {'name': 'Unknown', 'similarity': 0.0}
        
        embedding = embedding.flatten()
        similarities = cosine_similarity([embedding], np.array(self.known_face_embeddings))[0]
        max_idx = np.argmax(similarities)
        
        if max_idx >= len(self.known_names):
            return {'name': 'Unknown', 'similarity': 0.0}
            
        max_sim = similarities[max_idx]
        
        return {
            'name': self.known_names[max_idx] if max_sim >= self.FACE_THRESHOLD else 'Unknown',
            'similarity': max_sim
        }

    def _match_ear(self, embeddings: List[np.ndarray]) -> Dict:
        """Match ears against database"""
        if not self.known_ear_embeddings or not embeddings:
            return {'name': 'Unknown', 'similarity': 0.0}
            
        best_sim = -1
        best_idx = -1
        
        for emb in embeddings:
            similarities = cosine_similarity([emb], np.array(self.known_ear_embeddings))[0]
            current_max = np.max(similarities)
            if current_max > best_sim:
                best_sim = current_max
                best_idx = np.argmax(similarities)
        
        if best_idx == -1 or best_idx >= len(self.known_names):
            return {'name': 'Unknown', 'similarity': 0.0}
            
        return {
            'name': self.known_names[best_idx] if best_sim >= self.EAR_THRESHOLD else 'Unknown',
            'similarity': best_sim
        }

def main():
    # Initialize system with ear detection model
    system = MultimodalBiometricSystem("best_3.pt")  # Path to your YOLOv8 ear model
    
    while True:
        print("\n==== Multimodal Biometric System ====")
        print("1. Register New Person")
        print("2. Start Recognition")
        print("3. View Database")
        print("4. Exit")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter name to register: ").strip()
            if name:
                if system.register_person(name):
                    print(f"Successfully registered {name}!")
                else:
                    print(f"Failed to register {name}")
            else:
                print("Invalid name")
        
        elif choice == '2':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                continue
                
            print("Starting recognition... Press Q to quit")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, results = system.recognize_person(frame)
                cv2.imshow("Recognition", frame)
                
                for result in results:
                    if result['name'] == 'Unknown':
                        print("Unknown person detected")
                    else:
                        print(f"Recognized: {result['name']} ({result['confidence']*100:.1f}%)")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            print("\nRegistered Persons:")
            unique_names = list(set(system.known_names))
            for name in unique_names:
                print(f"- {name}")
            print(f"\nTotal: {len(unique_names)} persons")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()