"""
typhoon_hk.py

对香港附近的台风轨迹生成动画。优先从给定 CSV/URL 读取，如果未提供则使用内置合成示例数据。
台风以蓝色圆圈表示，颜色深浅对应强度（例如最大风速）。

用法例子：
    python typhoon_hk.py --save --out typhoon_hk.gif
    python typhoon_hk.py --csv path/to/track.csv --save --out typhoon_hk.gif

CSV 期望包含字段: time, lat, lon, wind_kts（或 wind_ms）

"""
import argparse
import csv
import io
import math
import urllib.request
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.use('Agg')

# 香港大致经纬度范围用于画面裁切
HK_LON_MIN, HK_LON_MAX = 113.5, 115.5
HK_LAT_MIN, HK_LAT_MAX = 21.5, 23.5


def download_csv(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = r.read().decode('utf-8')
        return data
    except Exception as e:
        print('Download failed:', e)
        return None


def parse_track_csv(text):
    # 尝试解析常见列名
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    rows = []
    for r in reader:
        try:
            time = r.get('time') or r.get('datetime') or r.get('timestamp')
            lat = float(r.get('lat') or r.get('latitude') or r.get('LAT') or r.get('LATITUDE'))
            lon = float(r.get('lon') or r.get('longitude') or r.get('LON') or r.get('LONGITUDE'))
            wind = r.get('wind_kts') or r.get('wind_kmh') or r.get('wind_ms') or r.get('vmax') or r.get('wind')
            if wind is None:
                wind_v = 0.0
            else:
                wind_v = float(wind)
            rows.append({'time': time, 'lat': lat, 'lon': lon, 'wind': wind_v})
        except Exception:
            continue
    return rows


def synthetic_track():
    # 生成一个向香港移动并靠近的合成台风路径（经纬度和强度）
    lons = np.linspace(120.0, 114.2, 20)
    lats = np.linspace(18.0, 22.3, 20)
    winds = np.linspace(40, 120, 20)  # 节
    times = [datetime(2025,9,18,0,0) + i * (datetime(2025,9,18,1,0) - datetime(2025,9,18,0,0)) for i in range(20)]
    rows = [{'time': t.isoformat(), 'lat': float(lat), 'lon': float(lon), 'wind': float(w)} for t, lat, lon, w in zip(times, lats, lons, winds)]
    return rows


def wind_to_color(winds, cmap=plt.get_cmap('Blues')):
    # 将风速映射到 0-1，并返回 RGBA
    mn, mx = np.min(winds), np.max(winds)
    if mx - mn < 1e-6:
        mx = mn + 1.0
    norm = Normalize(vmin=mn, vmax=mx)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    return mapper.to_rgba(winds), norm


def run(csv_path=None, url=None, save=False, out='typhoon_hk.gif', fps=3):
    rows = None
    if url:
        text = download_csv(url)
        if text:
            rows = parse_track_csv(text)
    if csv_path and rows is None:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                rows = parse_track_csv(f.read())
        except Exception as e:
            print('Failed to read csv:', e)
    if rows is None or len(rows) == 0:
        print('Using synthetic track (no data loaded)')
        rows = synthetic_track()

    # 过滤落在近海/中国南海范围内的点以聚焦香港附近（可选）
    # rows = [r for r in rows if HK_LON_MIN <= r['lon'] <= HK_LON_MAX and HK_LAT_MIN <= r['lat'] <= HK_LAT_MAX]

    lons = np.array([r['lon'] for r in rows])
    lats = np.array([r['lat'] for r in rows])
    winds = np.array([r['wind'] for r in rows])

    # 将风速统一单位（如果有怀疑，假设输入为节 kts）
    # 这里简单处理：若值看起来像 km/h（>300）则转为 kts
    winds = np.where(winds > 300, winds / 1.852 / 1.0, winds)

    colors, norm = wind_to_color(winds)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(HK_LON_MIN, HK_LON_MAX)
    ax.set_ylim(HK_LAT_MIN, HK_LAT_MAX)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Typhoon track near Hong Kong')
    ax.grid(True, linestyle='--', alpha=0.3)

    # 画香港点
    ax.scatter([114.1694], [22.3193], c='red', s=30, zorder=5)
    ax.text(114.18, 22.32, 'Hong Kong', color='red')

    # 初始点为空集合
    scat = ax.scatter([], [], s=[], c=[], cmap='Blues', vmin=np.min(winds), vmax=np.max(winds), zorder=6)
    trail, = ax.plot([], [], color='lightblue', linewidth=1, alpha=0.6)

    def animate(i):
        xi = lons[:i+1]
        yi = lats[:i+1]
        wi = winds[:i+1]
        rgba, _ = wind_to_color(wi)
        sizes = 50 * (wi / np.max(winds)) + 20
        scat.set_offsets(np.c_[xi, yi])
        scat.set_color(rgba)
        scat.set_sizes(sizes)
        trail.set_data(xi, yi)
        return scat, trail

    anim = FuncAnimation(fig, animate, frames=len(rows), interval=1000/fps, blit=True)

    if save:
        print('Saving', out)
        try:
            anim.save(out, writer='pillow', fps=fps)
            print('Saved', out)
        except Exception as e:
            print('Failed to save:', e)
    else:
        matplotlib.use('TkAgg')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='path to CSV with columns time,lat,lon,wind')
    parser.add_argument('--url', help='URL to download CSV')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--out', default='typhoon_hk.gif')
    parser.add_argument('--fps', type=int, default=2)
    args = parser.parse_args()
    run(csv_path=args.csv, url=args.url, save=args.save, out=args.out, fps=args.fps)
