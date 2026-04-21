import os
import sys
import cv2

# 1. 确保当前目录和模型仓库路径在 sys.path 中，优先级最高，加载当前本地的代码和模型
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
model_repo_path = os.path.join(root_dir, "face_recognition_models")
sys.path.insert(0, model_repo_path)

try:
    import face_recognition
    import face_recognition_models
    print("成功关联本地代码！")
    print(f"代码位置：{face_recognition.__file__}")
    print(f"模型位置：{face_recognition_models.__file__}")
except AttributeError as e:
    print(f"依然失败：{e}")
    print("目前的搜索路径列表：")
    for p in sys.path[:3] : print(f"    - {p}")

def run_face_locator():
    # 2. 初始化摄像头 (0 通常是内置摄像头)
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("[ERROR] 无法打开摄像头，请检查权限或设备连接")
        return

    print("[INFO] 引擎启动成功！正在从本地源码进行推理...")
    print("[INFO] 按下 'q' 键退出程序")

    # 性能参数：每秒处理帧数优化
    process_this_frame = True

    while True:
        # 读取一帧画面
        ret, frame = video_capture.read()
        if not ret:
            break

        # 3. 性能优化：将画面缩小到 1/4 尺寸，极大提升检测 FPS
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # 转换颜色空间 (OpenCV BGR -> face_recognition RGB)
        rgb_small_frame = small_frame[:, :, ::-1]

        # 隔帧处理：如果 CPU 较弱，可以开启此项（目前设为全量检测以保证丝滑）
        if process_this_frame:
            # 4. 调用本地仓库的定位接口
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # 5. 绘制检测结果
        for (top, right, bottom, left) in face_locations:
            # 将坐标还原（因为之前缩小了 4 倍）
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 画出定位框（亮绿色）
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # 在框上方加个酷炫的标签
            cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "HUMAN FACE", (left + 6, top - 10), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # 6. 显示实时结果
        cv2.imshow('Face Detection Project - Live Mode', frame)

        # 退出机制
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_locator()
