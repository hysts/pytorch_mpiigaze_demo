import cv2
from collections import Counter

def smooth_predictions(predictions, video, window_size=10):
    with open(predictions, 'r') as f:
        preds = f.readlines()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video[:-4]}_smoothed.avi', fourcc, fps, (int(cap.get(3)),int(cap.get(4))))

    smoothed_preds = ['' for _ in range(len(preds))]
    for i in range(len(preds) - window_size):
        window = preds[i:i + window_size]
        most_common = Counter(window).most_common(1)[0][0]
        smoothed_preds[i+ window_size//2] = most_common
    
    front_fill = back_fill = ''
    index = 0
    while smoothed_preds[index] == '':
        index += 1
    front_fill = smoothed_preds[index]
    for i in range(index):
        smoothed_preds[i] = front_fill
    
    index = len(smoothed_preds) - 1
    while smoothed_preds[index] == '':
        index -= 1
    back_fill = smoothed_preds[index]
    for i in range(len(smoothed_preds) - 1, index, -1):
        smoothed_preds[i] = back_fill

    # print(len(smoothed_preds), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(smoothed_preds)
    # assert len(smoothed_preds) == len(preds)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= len(smoothed_preds):
            print(f'breaking after {count} frames')
            break

        cv2.putText(frame, f"Smoothed Gaze: {smoothed_preds[count].strip()}", (150, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
        count += 1
        out.write(frame)

    cap.release()
    out.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    smooth_predictions('assets/inputs/gazeEstimation4.txt', 'assets/results/gazeEstimation4.mp4')