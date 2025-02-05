import cv2
import argparse
from ultralytics import YOLO
import time

def detect_image(image_path, model):
    image = cv2.imread(image_path)
    results = model.predict(image)
    annotated_image = results[0].plot()
    
    # 결과 저장 및 표시
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated_image)
    # cv2.imshow("YOLO Image Detection", annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f"저장된 이미지: {output_path}")

def detect_webcam(model):
    cap = cv2.VideoCapture(2)  # 웹캠 번호 (기본 웹캠: 0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # YOLO 객체 감지 (3초마다 수행)
        start_time = time.time()
        results = model.predict(frame, show=False)
        annotated_frame = results[0].plot()

        # 화면에 표시
        cv2.imshow("YOLO Real-time Detection (Every 3 seconds)", annotated_frame)

        # ESC 키(27번) 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # # 3초 대기
        # time.sleep(1 - ((time.time() - start_time) % 1))

    cap.release()
    cv2.destroyAllWindows()
    
def main():
    """ argparse를 사용하여 모드 선택 """
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--mode", type=str, required=True, choices=["image", "webcam"], help="실행 모드 선택 (image 또는 webcam)")
    parser.add_argument("--image", type=str, help="이미지 감지 모드에서 사용할 이미지 경로")

    args = parser.parse_args()
    model = YOLO("yolo11n.pt")  # YOLO 모델 로드

    if args.mode == "image":
        if not args.image:
            print("이미지 경로를 입력하세요. 예: --mode image --image input.jpg")
            return
        detect_image(args.image, model)
    
    elif args.mode == "webcam":
        detect_webcam(model)

if __name__ == "__main__":
    main()
