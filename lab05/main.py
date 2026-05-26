import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.utils import flow_to_image

def get_ground_mask(flow, frame_shape):
    h, w = frame_shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 1. Создаем базовую зону "земли" (нижние 45% кадра)
    ground_zone = np.zeros((h, w), dtype=np.uint8)
    ground_zone[int(h * 0.55):, :] = 255
    
    # 2. Определяем точку схода (упрощенно - центр горизонта)
    foe = np.array([w // 2, h * 0.45])
    
    # Создаем сетку координат
    yy, xx = np.mgrid[:h, :w]
    # Вектор от FOE до каждой точки пикселя
    dx = xx - foe[0]
    dy = yy - foe[1]
    _, expected_ang = cv2.cartToPolar(dx, dy)
    
    # 3. Сравниваем реальный угол потока с ожидаемым (для плоскости земли)
    # Дорога должна двигаться примерно по лучам из FOE
    ang_diff = np.abs(ang - expected_ang)
    ang_diff = np.minimum(ang_diff, 2*np.pi - ang_diff)
    
    # Условие: правильный угол + достаточная скорость + нижняя часть
    mask = (ang_diff < 0.5) & (mag > 1.2) & (ground_zone > 0)
    
    # 4. Финальная шлифовка (убираем шум)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.GaussianBlur(mask_uint8, (15, 15), 0)
    
    return mask_uint8

def process_final_v2(input_path, factor=8):
    device = "cpu"
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=True).to(device).eval()

    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(cv2.CAP_PROP_FPS) or 30), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_slow = cv2.VideoWriter("1_RAFT_SlowMo_8x.mp4", fourcc, fps, h)
    out_nav = cv2.VideoWriter("2_Navigator_Fixed.mp4", fourcc, fps, h)
    out_ground = cv2.VideoWriter("3_Ground_Detection.mp4", fourcc, fps, h)
    out_raft_color = cv2.VideoWriter("4_RAFT_Flow_Color.mp4", fourcc, fps, h)

    ret, frame1 = cap.read()
    grid_y, grid_x = np.mgrid[0:h[1], 0:h[0]].astype(np.float32)
    count = 0

    try:
        while True:
            ret, frame2 = cap.read()
            if not ret: break
            
            # RAFT Inference
            proc_size = (440, 248) # Пропорции для скорости на CPU
            t1 = torch.from_numpy(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0)
            t2 = torch.from_numpy(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0)
            img1_p = F.resize(t1, proc_size, antialias=False)
            img2_p = F.resize(t2, proc_size, antialias=False)
            img1_p, img2_p = transforms(img1_p, img2_p)

            with torch.no_grad():
                flow_f_raw = model(img1_p, img2_p)[-1][0]
                flow_b_raw = model(img2_p, img1_p)[-1][0]

            # Подготовка потока для отрисовки
            flow_f_np = flow_f_raw.permute(1, 2, 0).numpy()
            full_f = cv2.resize(flow_f_np, h)
            full_f[:,:,0] *= (h[0]/proc_size[1]); full_f[:,:,1] *= (h[1]/proc_size[0])
            
            full_b = cv2.resize(flow_b_raw.permute(1, 2, 0).numpy(), h)
            full_b[:,:,0] *= (h[0]/proc_size[1]); full_b[:,:,1] *= (h[1]/proc_size[0])

            # --- 1. RAFT Flow Color (Цветное как у товарища) ---
            flow_color = flow_to_image(flow_f_raw).permute(1, 2, 0).numpy()
            flow_color_bgr = cv2.resize(cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR), h)
            out_raft_color.write(flow_color_bgr)

            # --- 2. Navigator (Инвертированная стрелка) ---
            nav_frame = frame1.copy()
            # Берем среднее движение, но инвертируем его (-), чтобы показать движение КАМЕРЫ
            avg_move = -np.mean(full_f, axis=(0, 1)) * 10 
            center = (h[0] // 2, h[1] // 2)
            endpoint = (int(center[0] + avg_move[0]), int(center[1] + avg_move[1]))
            cv2.arrowedLine(nav_frame, center, endpoint, (0, 255, 0), 8, tipLength=0.2)
            cv2.putText(nav_frame, "MOVE DIR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out_nav.write(nav_frame)

            # --- 3. Ground Detection (Улучшенная маска) ---
            m = get_ground_mask(full_f, h)
            ground_vis = frame1.copy()
            ground_vis[m > 128] = [0, 255, 0] # Зеленый оверлей
            out_ground.write(cv2.addWeighted(frame1, 0.7, ground_vis, 0.3, 0)) # Умножаем на 0.2 и добавляем в frame1

            # --- 4. Slow Motion 8x ---
            for i in range(factor):
                alpha = i / factor
                m_fx, m_fy = grid_x - alpha * full_f[:,:,0], grid_y - alpha * full_f[:,:,1]
                w_f = cv2.remap(frame1, m_fx, m_fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                m_bx, m_by = grid_x - (1 - alpha) * full_b[:,:,0], grid_y - (1 - alpha) * full_b[:,:,1]
                w_b = cv2.remap(frame2, m_bx, m_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out_slow.write(cv2.addWeighted(w_f, 1.0 - alpha, w_b, alpha, 0))

            frame1 = frame2
            count += 1
            if count % 10 == 0: print(f"Processing: {count}/{total_frames}")

    finally:
        for v in [out_slow, out_nav, out_ground, out_raft_color]: v.release()
        cap.release()
        print("Готово! Все 4 файла обновлены.")

if __name__ == "__main__":
    process_final_v2("video_1.mp4", factor=8)
