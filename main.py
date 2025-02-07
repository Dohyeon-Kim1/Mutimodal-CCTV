import requests
import threading
import cv2
from ultralytics import YOLO


current_frames = []
current_caption = ""


def post_frames():
    global current_caption
    
    frames = current_frames[0]
    response = requests.post(
        "http://127.0.0.1:5001/receive",
        json={"frames": frames}
    )
    current_caption = response.json()["caption"]

    threading.Timer(4, post_frames).start()


def main():
    yolo = YOLO("yolo_cctv.pt")
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    processed_frames = 0
    max_ref_frame = 8
    
    post_frames()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        else:
            processed_frames += 1
        
        if processed_frames % 5 == 0:
            if len(current_frames) == max_ref_frame:
                current_frames.pop(0)
            current_frames.append(frame)
        
        results = yolo.predict(frame, imgsz=640, augment=False, show=False)
        annotated_frame = results[0].plot()

        # 화면에 표시
        cv2.imshow("YOLO Real-time Detection", annotated_frame)
        print(current_caption)

        # ESC 키(27번) 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()


