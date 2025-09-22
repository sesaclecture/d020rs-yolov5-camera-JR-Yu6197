import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames

while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # TODO: break the loop on error
    if ret is None:
        break

    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    # TODO: Boudning box 그리기
    


    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음 
        x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        confidence = float(obj[4])
        class_id = int(obj[5])
        class_name = model.names[class_id]
        label = f'{class_name} {confidence:.2f}' # 클래스 이름과 정확도(소수점 2자리)

        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        cv2.rectangle(
            frame,          # 원본 이미지
            (x1, y1),       # 시작 좌표 (왼쪽 위)
            (x2, y2),       # 종료 좌표 (오른쪽 아래)
            (255, 0, 0),    # 색상 (BGR 순서, 파란색)
            2               # 선 두께
        )
        cv2.putText(
            frame,          # 원본 이미지
            label,          # 표시할 텍스트
            (x1, y1 - 10),  # 텍스트 시작 좌표 (사각형 위쪽)
            cv2.FONT_HERSHEY_SIMPLEX, # 폰트
            0.9,            # 폰트 크기
            (255, 0, 0),    # 폰트 색상 (파란색)
            2               # 폰트 두께
        )
        obj_info = list(map(int, obj))
        print(f"Object {i}: {model.names[obj_info[5]]}")

    # TODO: 화면 표시
    cv2.imshow('Yolo Camera preview', frame)

    # TODO: 종료를 위한 key 처리
    if cv2.waitKey(1) & 0xff == 27: # 27: Esc key
        break
        

cap.release()
cv2.destroyAllWindows()
