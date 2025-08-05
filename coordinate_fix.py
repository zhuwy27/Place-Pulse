import pandas as pd
import numpy as np
import requests
import random
import math
import ast
import os
import pickle
import glob
from time import sleep
import re
import folium
from collections import defaultdict
from branca.element import Template, MacroElement

# ========== 坐标系转换函数 (WGS84 <-> GCJ02) 优化版 ==========
PI = math.pi
AXIS = 6378245.0
OFFSET = 0.00669342162296594323  # (a^2 - b^2) / a^2


def wgs84_to_gcj02(lon, lat):
    """将WGS84坐标转换为GCJ02坐标（火星坐标系）"""
    # 确保输入是浮点数
    try:
        lon = float(lon)
        lat = float(lat)
    except (ValueError, TypeError):
        print(f"❌ 无效的坐标值: lon={lon}, lat={lat}")
        return lon, lat

    if _out_of_china(lon, lat):
        return lon, lat

    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)

    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - OFFSET * magic * magic
    sqrtmagic = math.sqrt(magic)

    dlat = (dlat * 180.0) / ((AXIS * (1 - OFFSET)) / (magic * sqrtmagic) * PI)
    dlon = (dlon * 180.0) / (AXIS / sqrtmagic * math.cos(radlat) * PI)

    result_lon = lon + dlon
    result_lat = lat + dlat
    return result_lon, result_lat


def gcj02_to_wgs84(lon, lat):
    """将GCJ02坐标转换为WGS84坐标 - 优化版，减少误差累积"""
    # 确保输入是浮点数
    try:
        lon = float(lon)
        lat = float(lat)
    except (ValueError, TypeError):
        print(f"❌ 无效的坐标值: lon={lon}, lat={lat}")
        return lon, lat

    if _out_of_china(lon, lat):
        return lon, lat

    # 迭代计算以获得更精确的结果
    threshold = 1e-7  # 收敛阈值
    wgs_lon, wgs_lat = lon, lat
    for _ in range(5):  # 最多迭代5次
        gcj_lon, gcj_lat = wgs84_to_gcj02(wgs_lon, wgs_lat)
        dlon = gcj_lon - lon
        dlat = gcj_lat - lat
        wgs_lon -= dlon
        wgs_lat -= dlat
        if abs(dlon) < threshold and abs(dlat) < threshold:
            break

    return wgs_lon, wgs_lat


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
    return ret


def _out_of_china(lon, lat):
    """判断坐标是否在中国境外（境外坐标不进行偏移）"""
    return not (73.66 <= lon <= 135.05 and 18.16 <= lat <= 53.55)


# ========== 修复后的高德API调用函数 ==========
def get_walk_time_api_fixed(origin, destination, gaode_api_key):
    """修复后的API调用函数，确保坐标系转换正确"""
    # 增加坐标验证
    if not all(k in origin for k in ['lon', 'lat']) or not all(k in destination for k in ['lon', 'lat']):
        print("❌ 坐标数据不完整，缺少经纬度信息")
        return 30.0, []

    # 验证原始坐标是否合理
    try:
        origin_lon, origin_lat = float(origin['lon']), float(origin['lat'])
        dest_lon, dest_lat = float(destination['lon']), float(destination['lat'])
        
        print(f"📌 原始WGS84坐标 - 起点: ({origin_lon:.6f}, {origin_lat:.6f}), 终点: ({dest_lon:.6f}, {dest_lat:.6f})")
        
    except:
        print("❌ 坐标格式错误，无法验证")
        return 30.0, []

    # ⚠️ 关键修复：将WGS84坐标转换为GCJ02坐标，用于高德API
    origin_gcj_lon, origin_gcj_lat = wgs84_to_gcj02(origin_lon, origin_lat)
    dest_gcj_lon, dest_gcj_lat = wgs84_to_gcj02(dest_lon, dest_lat)
    
    print(f"🔄 转换后GCJ02坐标 - 起点: ({origin_gcj_lon:.6f}, {origin_gcj_lat:.6f}), 终点: ({dest_gcj_lon:.6f}, {dest_gcj_lat:.6f})")

    url = 'https://restapi.amap.com/v3/direction/driving'
    params = {
        'origin': f"{origin_gcj_lon:.6f},{origin_gcj_lat:.6f}",
        'destination': f"{dest_gcj_lon:.6f},{dest_gcj_lat:.6f}",
        'key': gaode_api_key,
        'extensions': 'all',
        'strategy': 0,
        'output': 'json'
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data['status'] == '1' and data['route']['paths']:
                    path = data['route']['paths'][0]
                    time_cost = int(path['duration']) / 60

                    # ⚠️ 关键修复：提取路径坐标并转换回WGS84用于folium可视化
                    coords = []
                    steps = path['steps']

                    for step in steps:
                        if 'polyline' in step and step['polyline']:
                            polyline_str = step['polyline'].strip()
                            if polyline_str:
                                for pair in polyline_str.split(';'):
                                    if pair.strip():
                                        try:
                                            gcj_lon, gcj_lat = map(float, pair.split(','))
                                            # ⚠️ 关键修复：API返回的是GCJ02坐标，需要转换回WGS84供folium使用
                                            wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
                                            if -180 <= wgs_lon <= 180 and -90 <= wgs_lat <= 90:
                                                coords.append((wgs_lat, wgs_lon))  # folium需要(lat, lon)格式
                                        except (ValueError, IndexError):
                                            continue

                    # 确保起点和终点坐标在路径中（使用原始WGS84坐标）
                    if coords:
                        # 验证起点是否匹配（使用原始WGS84坐标）
                        start_diff = math.hypot(coords[0][1] - origin_lon, coords[0][0] - origin_lat)
                        if start_diff > 0.001:  # 约100米差异
                            coords.insert(0, (origin_lat, origin_lon))
                            print(f"⚠️ 路径起点不匹配，已修正 (差异: {start_diff:.6f})")

                        # 验证终点是否匹配（使用原始WGS84坐标）
                        end_diff = math.hypot(coords[-1][1] - dest_lon, coords[-1][0] - dest_lat)
                        if end_diff > 0.001:  # 约100米差异
                            coords.append((dest_lat, dest_lon))
                            print(f"⚠️ 路径终点不匹配，已修正 (差异: {end_diff:.6f})")
                    else:
                        # 生成直线路径，使用原始WGS84坐标
                        coords = [
                            (origin_lat, origin_lon),
                            (dest_lat, dest_lon)
                        ]
                        print("⚠️ 使用直线路径连接景点（WGS84坐标）")

                    return time_cost, coords
                else:
                    error_info = data.get('info', '未知错误')
                    print(f"⚠️ API返回状态错误: {error_info}，尝试次数: {attempt + 1}")
                    if attempt < max_retries - 1:
                        sleep(2.0)
                        continue
            else:
                print(f"⚠️ HTTP状态码: {r.status_code}，尝试次数: {attempt + 1}")
                if attempt < max_retries - 1:
                    sleep(1.0)
                    continue
        except Exception as e:
            print(f"⚠️ API调用异常: {e}，尝试次数: {attempt + 1}")
            if attempt < max_retries - 1:
                sleep(1.0)
                continue

    # 所有重试失败，使用直线路径（WGS84坐标）
    print(f"❌ 所有重试都失败，使用直线路径连接景点（WGS84坐标）")
    coords = [
        (origin_lat, origin_lon),
        (dest_lat, dest_lon)
    ]
    # 计算直线距离
    lon_diff = math.radians(dest_lon - origin_lon)
    lat_diff = math.radians(dest_lat - origin_lat)
    a = math.sin(lat_diff / 2) ** 2 + math.cos(math.radians(origin_lat)) * math.cos(
        math.radians(dest_lat)) * math.sin(lon_diff / 2) ** 2
    distance_km = 6371 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    time_cost = distance_km * 10  # 假设平均时速6km/h，转换为分钟
    return time_cost, coords


# ========== 修复后的可视化函数 ==========
def plot_routes_fixed(df, routes, route_cache, output_html_path):
    """修复后的可视化函数，确保景点坐标正确显示"""
    
    print("🗺️ 开始生成地图可视化...")
    
    # ⚠️ 关键修复：收集所有景点的WGS84坐标（不进行任何转换）
    all_attraction_coords = []
    for idx, row in df.iterrows():
        try:
            # 直接使用原始WGS84坐标，不进行转换
            lat, lon = float(row['lat']), float(row['lon'])
            all_attraction_coords.append((lat, lon))
            print(f"景点 {row['name']}: WGS84坐标 ({lat:.6f}, {lon:.6f})")
        except (ValueError, TypeError):
            print(f"❌ 无效坐标: {row['name']}")
            continue
    
    # 计算地图中心点
    if all_attraction_coords:
        lats, lons = zip(*all_attraction_coords)
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        print(f"🗺️ 地图中心点: ({center_lat:.6f}, {center_lon:.6f})")
    else:
        center_lat, center_lon = 23.1291, 113.2644  # 广州中心坐标
    
    # ⚠️ 关键修复：使用OpenStreetMap底图，它使用WGS84坐标系
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'  # WGS84坐标系底图
    )
    
    # ⚠️ 关键修复：添加所有景点标记，使用原始WGS84坐标
    attraction_markers = {}
    for idx, row in df.iterrows():
        try:
            # 直接使用原始WGS84坐标，不进行任何转换
            lat, lon = float(row['lat']), float(row['lon'])
            name = row['name']
            attraction_markers[idx] = (lat, lon)
            
            # 添加景点标记
            folium.CircleMarker(
                location=[lat, lon],  # 直接使用WGS84坐标
                radius=8,
                color='darkblue',
                fill=True,
                fill_color='darkblue',
                fill_opacity=0.7,
                popup=f"{name}<br>WGS84坐标: ({lat:.6f}, {lon:.6f})"
            ).add_to(m)
            
            # 添加文字标签
            folium.map.Marker(
                [lat + 0.0005, lon],  # 微调标签位置
                icon=folium.DivIcon(
                    html=f'<div style="font-size:12px; color:darkblue; font-weight:bold;">{name}</div>'
                )
            ).add_to(m)
            
            print(f"✅ 添加景点标记: {name} at WGS84 ({lat:.6f}, {lon:.6f})")
            
        except Exception as e:
            print(f"⚠️ 添加景点标记失败: {row.get('name', '未知')}, 错误: {e}")
    
    # 绘制路线
    color_dict = {'6h': '#e74c3c', '12h': '#2ecc71'}
    
    for route_name, points in routes.items():
        print(f"\n🗺️ 绘制 {route_name} 路线...")
        complete_route = []
        
        for i in range(len(points) - 1):
            current_order, current_name, (current_lon, current_lat) = points[i]
            next_order, next_name, (next_lon, next_lat) = points[i + 1]
            
            # ⚠️ 关键修复：使用WGS84坐标进行路径连接
            current_lat, current_lon = float(current_lat), float(current_lon)
            next_lat, next_lon = float(next_lat), float(next_lon)
            
            # 找到对应的索引
            current_idx = None
            next_idx = None
            for idx, (marker_lat, marker_lon) in attraction_markers.items():
                if abs(marker_lat - current_lat) < 0.00001 and abs(marker_lon - current_lon) < 0.00001:
                    current_idx = idx
                if abs(marker_lat - next_lat) < 0.00001 and abs(marker_lon - next_lon) < 0.00001:
                    next_idx = idx
                if current_idx is not None and next_idx is not None:
                    break
            
            if current_idx is None or next_idx is None:
                print(f"⚠️ 无法找到景点索引 - 当前: {current_name}, 下一个: {next_name}")
                segment_coords = [(current_lat, current_lon), (next_lat, next_lon)]
            else:
                # 使用缓存的路径（应该是WGS84坐标）
                key = f"{current_idx}_{next_idx}"
                segment_coords = route_cache.get(key, [])
                
                if not segment_coords:
                    # 生成直线路径
                    segment_coords = [(current_lat, current_lon), (next_lat, next_lon)]
                    print(f"⚠️ 生成直线路径: {current_name} -> {next_name}")
            
            # 添加到完整路线
            if complete_route and segment_coords:
                if complete_route[-1] == segment_coords[0]:
                    complete_route.extend(segment_coords[1:])
                else:
                    complete_route.extend(segment_coords)
            elif segment_coords:
                complete_route.extend(segment_coords)
        
        # 绘制完整路线
        if complete_route:
            folium.PolyLine(
                complete_route,
                color=color_dict.get(route_name, 'gray'),
                weight=5,
                opacity=0.8,
                popup=f"{route_name}路线"
            ).add_to(m)
            print(f"✅ {route_name} 路线绘制完成，{len(complete_route)}个坐标点")
    
    # 添加图例
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed; 
        bottom: 30px; right: 30px; width: 220px; 
        z-index:9999; font-size:14px;
        background-color: white;
        border:2px solid grey;
        padding: 10px;
    ">
    <b>图例 (WGS84坐标系)</b><br>
    <span style="color:darkblue;">●</span> 景点位置<br>
    {'<br>'.join([f'<span style="color:{color_dict.get(lbl, "gray")};">━━━</span> {lbl}路线' for lbl in routes.keys()])}
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)
    
    m.save(output_html_path)
    print(f"✅ 地图已保存为 {output_html_path}")


# ========== 坐标系测试函数 ==========
def test_coordinate_conversion():
    """测试坐标系转换的正确性"""
    print("🧪 测试坐标系转换...")
    
    # 广州塔的真实坐标（WGS84）
    test_wgs84_lon, test_wgs84_lat = 113.3191, 23.1098
    print(f"原始WGS84坐标: ({test_wgs84_lon}, {test_wgs84_lat})")
    
    # 转换为GCJ02
    gcj02_lon, gcj02_lat = wgs84_to_gcj02(test_wgs84_lon, test_wgs84_lat)
    print(f"转换后GCJ02坐标: ({gcj02_lon}, {gcj02_lat})")
    
    # 再转换回WGS84
    converted_back_lon, converted_back_lat = gcj02_to_wgs84(gcj02_lon, gcj02_lat)
    print(f"转换回WGS84坐标: ({converted_back_lon}, {converted_back_lat})")
    
    # 计算误差
    error_lon = abs(converted_back_lon - test_wgs84_lon)
    error_lat = abs(converted_back_lat - test_wgs84_lat)
    print(f"转换误差: 经度 {error_lon:.8f}, 纬度 {error_lat:.8f}")
    
    if error_lon < 1e-6 and error_lat < 1e-6:
        print("✅ 坐标系转换正确")
    else:
        print("❌ 坐标系转换存在误差")


if __name__ == '__main__':
    # 运行坐标系测试
    test_coordinate_conversion()