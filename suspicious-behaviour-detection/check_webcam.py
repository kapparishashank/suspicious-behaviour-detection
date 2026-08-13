import cv2

def check_webcam():
    print("Checking webcam availability...")
    # Try indices 0 to 5
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam index {i} is available.")
            ret, frame = cap.read()
            if ret:
                print(f"  - Successfully read a frame from index {i}. Frame shape: {frame.shape}")
            else:
                print(f"  - Could not read a frame from index {i}.")
            cap.release()
        else:
            print(f"Webcam index {i} is not available.")

if __name__ == "__main__":
    check_webcam()
