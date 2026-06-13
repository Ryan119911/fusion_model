import json
import csv
from pathlib import Path

def generate_fake_trajectories(graphics_path, out_csv_path):
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 严格对齐代码库中 trajectory_dataset.py 要求的表头
    fieldnames = ["character", "sample_id", "stroke_id", "point_id", "x", "y", "z", "alpha", "beta", "gamma", "state"]

    print(f"正在读取 MakeMeAHanzi 数据: {graphics_path}")
    try:
        with open(graphics_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {graphics_path}。请确保数据已放入正确位置。")
        return

    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for line in lines:
            if not line.strip(): continue
            data = json.loads(line)
            char = data.get('character', '')
            medians = data.get('medians', [])

            # 伪造一个唯一的样本ID
            sample_id = f"{char}_fake_sim"

            for stroke_idx, median in enumerate(medians):
                num_points = len(median)
                if num_points == 0: continue

                for pt_idx, pt in enumerate(median):
                    x, y = pt[0], pt[1]

                    # --- 核心：捏造 Z 轴（下压深度）曲线 ---
                    # 书法规律：起笔重（深），行笔轻（浅），收笔重（深）
                    t = pt_idx / max(1, (num_points - 1))
                    
                    # 构造一个抛物线深度的物理模型：两端 Z=4.0，中间 Z=2.0
                    z = 4.0 - 8.0 * (t - 0.5)**2
                    z = max(1.0, z) # 保证始终有下压

                    # 判定笔画状态：0=落笔(DOWN), 1=行笔(MOVE), 2=提笔(UP)
                    if pt_idx == 0:
                        state = 0 
                    elif pt_idx == num_points - 1:
                        state = 2
                        z = 0.0 # 提笔时深度为0
                    else:
                        state = 1

                    # 笔姿态角度（默认毛笔垂直纸面，因此全部给0）
                    alpha, beta, gamma = 0.0, 0.0, 0.0

                    # 写入伪造的轨迹点
                    writer.writerow({
                        "character": char,
                        "sample_id": sample_id,
                        "stroke_id": stroke_idx,
                        "point_id": pt_idx,
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "z": round(z, 2),
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "state": state
                    })
            count += 1
            if count % 1000 == 0:
                print(f"已伪造 {count} 个汉字的 3D 轨迹...")

    print(f"\n大功告成！已成功伪造 {count} 个汉字的物理轨迹。")
    print(f"文件已保存至: {out_csv_path}")

if __name__ == "__main__":
    # 配置输入与输出路径
    GRAPHICS_TXT = "data/raw/makemeahanzi/graphics.txt"
    OUT_CSV = "data/raw/trajectories.csv"
    
    generate_fake_trajectories(GRAPHICS_TXT, OUT_CSV)