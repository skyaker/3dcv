import os
import cv2

if __name__ == "__main__":
    dir = "/Users/jungdongwook/code/3DCV/agisoft_tools/"
    FRAME_STEP = 5

    os.makedirs(os.path.join(dir, "images_new"), exist_ok=True)

    files = os.listdir(dir)
    files = [f for f in files if f.find("MOV") > 0]

    count = 1
    for file in files:
        path = os.path.abspath(os.path.join(dir, file))
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        print("---------------->")

        while True:
            ret, im = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_STEP == 0:
                im_name = os.path.join(dir, "images_new", f"{count:05d}.png")
                cv2.imwrite(im_name, im)
                count += 1
            frame_idx += 1

        cap.release()
    print(f"Готово: {count-1} кадров")
