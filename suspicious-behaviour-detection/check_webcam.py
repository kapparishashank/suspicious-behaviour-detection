import cv2
from typing import List

def check_webcams(max_index: int = 5, stop_at_first: bool = False) -> List[int]:
    """
    Checks for available webcams and their ability to capture frames.
    
    Args:
        max_index (int): The maximum camera index to check (inclusive). Defaults to 5.
        stop_at_first (bool): If True, stops searching as soon as one working webcam is found. Defaults to False.
        
    Returns:
        List[int]: A list of working camera indices.
    """
    print("🔍 Checking webcam availability...")
    available_cameras = []

    for i in range(max_index + 1):  # +1 makes it inclusive (e.g., 0 to 5)
        try:
            cap = cv2.VideoCapture(i)
            
            # Check if the camera opened successfully
            if not cap.isOpened():
                continue  # Silently skip if not available

            # Try to read a frame to prove the camera actually works
            ret, frame = cap.read()
            cap.release()  # ALWAYS release the camera immediately!

            if ret and frame is not None:
                # Extract width and height for better info
                h, w = frame.shape[:2]
                print(f"✅ Webcam index {i} is working. (Resolution: {w}x{h})")
                available_cameras.append(i)
                
                if stop_at_first:
                    print("🛑 Stopped at first working camera (stop_at_first=True).")
                    break
            else:
                print(f"⚠️ Index {i} opened but couldn't read frames (virtual/broken camera).")

        except Exception as e:
            print(f"❌ Error accessing webcam index {i}: {e}")

    if not available_cameras:
        print("🚫 No working webcams found on your system.")
        
    return available_cameras


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set stop_at_first=True if you just want to find one working camera quickly
    working_cams = check_webcams(max_index=5, stop_at_first=True)
    
    # Example of how to use this in your main project
    if working_cams:
        best_cam = working_cams[0]
        print(f"\n📹 Recommendation: Use camera index {best_cam} for your detection script.")