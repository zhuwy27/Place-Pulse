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


# ========== 坐标验证与修正工具函数 ==========
def validate_and_correct_coordinates(df):
    """验证并修正DataFrame中的坐标数据，新增自动修正功能"""
    invalid_indices = []
    corrected_count = 0

    for idx, row in df.iterrows():
        try:
            lon, lat = float(row['lon']), float(row['lat'])

            # 中国大致经纬度范围验证
            if not (73 <= lon <= 136 and 18 <= lat <= 54):
                invalid_indices.append(idx)
                print(f"⚠️ 无效坐标: {row['name']} - ({lon}, {lat})，超出中国范围")
                continue

            # 检查是否在河流区域，如果是则尝试修正
            if is_point_in_river(lon, lat):
                print(f"⚠️ 坐标位于河流区域: {row['name']} - ({lon}, {lat})")

                # 尝试基于周边景点进行修正
                nearby_correction = _find_nearby_correction(lon, lat, df, exclude_idx=idx)
                if nearby_correction:
                    corrected_lon, corrected_lat = nearby_correction
                    df.at[idx, 'lon'] = corrected_lon
                    df.at[idx, 'lat'] = corrected_lat
                    print(f"✅ 已自动修正为: ({corrected_lon:.6f}, {corrected_lat:.6f})")
                    corrected_count += 1
                else:
                    print(f"⚠️ 无法自动修正，标记为可能错误")

        except (ValueError, TypeError):
            invalid_indices.append(idx)
            print(f"⚠️ 坐标格式错误: {row['name']}")

    print(f"坐标验证完成: 共 {len(df)} 个景点，{len(invalid_indices)} 个无效坐标，{corrected_count} 个已修正")
    return invalid_indices


def _find_nearby_correction(lon, lat, df, exclude_idx=None, radius_km=1):
    """寻找附近合理的坐标作为修正参考"""
    valid_points = []

    # 计算经纬度每度对应的公里数（约值）
    lon_per_km = 1 / 111.0  # 经度每度约111公里
    lat_per_km = 1 / (111.0 * math.cos(math.radians(lat)))  # 纬度每度公里数

    for idx, row in df.iterrows():
        if idx == exclude_idx:
            continue

        try:
            r_lon, r_lat = float(row['lon']), float(row['lat'])
            # 检查该点是否合理（不在河流中）
            if not is_point_in_river(r_lon, r_lat):
                # 计算距离
                lon_diff = abs(r_lon - lon)
                lat_diff = abs(r_lat - lat)

                # 转换为公里
                distance_km = math.sqrt(
                    (lon_diff / lon_per_km) ** 2 +
                    (lat_diff / lat_per_km) ** 2
                )

                if distance_km <= radius_km:
                    valid_points.append((r_lon, r_lat))
        except:
            continue

    if valid_points:
        # 返回附近有效点的平均值
        avg_lon = sum(p[0] for p in valid_points) / len(valid_points)
        avg_lat = sum(p[1] for p in valid_points) / len(valid_points)
        return avg_lon, avg_lat
    return None


def is_point_in_river(lon, lat, river_bounds=None):
    """判断点是否在河流范围内（更精确的判断）"""
    if river_bounds is None:
        river_bounds = {
            'min_lon': 113.25, 'max_lon': 113.35,
            'min_lat': 22.45, 'max_lat': 23.10
        }

    # 更精确的河流区域判断
    in_river = (river_bounds['min_lon'] < lon < river_bounds['max_lon'] and
                river_bounds['min_lat'] < lat < river_bounds['max_lat'])

    # 排除已知的河岸景点区域
    shore_areas = [
        # 区域1: 排除某些河岸区域
        {'min_lon': 113.28, 'max_lon': 113.32, 'min_lat': 23.10, 'max_lat': 23.13},
        # 区域2: 另一个河岸区域
        {'min_lon': 113.26, 'max_lon': 113.29, 'min_lat': 23.07, 'max_lat': 23.09}
    ]

    for area in shore_areas:
        if (area['min_lon'] < lon < area['max_lon'] and
                area['min_lat'] < lat < area['max_lat']):
            in_river = False
            break

    return in_river


# ========== 参数配置 ==========
# --- 文件与API ---
GAODE_API_KEY = 'b5b76bb1f365e6b3d9f86f09f7fe3041'  # 请替换为您的高德API密钥
CACHE_FILE_PREFIX = 'time_matrix_cache'  # 缓存文件前缀

# --- 蚁群算法 (ACO) ---
ANT_COUNT = 30
ACO_ITER = 100
ALPHA = 1
BETA = 5
RHO = 0.1
Q = 1

# --- 路线规划 ---
TIME_LIMITS = [360, 720]  # 6小时, 12小时 (分钟)
SLEEP_INTERVAL = 0.01
OUTPUT_ROUTE_FILE_SUFFIX = '_optimal_routes.xlsx'

# --- AHP 权重设定 ---
L1_TO_L2_IMPORTANCE = 5
OBJECTIVE_CRITERIA = ['cover', 'satisfaction', 'stay_ratio', 'diversity']
OBJECTIVE_COMPARISON_MATRIX = np.array([
    [1.0, 3.0, 5.0, 2.0],  # cover (行)
    [1 / 3.0, 1.0, 3.0, 1 / 2.0],  # satisfaction (行)
    [1 / 5.0, 1 / 3.0, 1.0, 1 / 4.0],  # stay_ratio (行)
    [1 / 2.0, 2.0, 4.0, 1.0]  # diversity (行)
])
BASE_MULTIPLIER = 20.0

# --- 可视化配置 ---
color_dict = {
    '6h': '#e74c3c',  # 鲜艳红色
    '12h': '#2ecc71'  # 鲜艳绿色
}


# ========== AHP 权重计算函数 ==========
def calculate_ahp_2x2_weights(importance_scale: int) -> dict:
    """使用AHP科学计算2x2矩阵的多样性权重"""
    if not 1 <= importance_scale <= 9:
        raise ValueError("重要性标度必须在1到9之间。")
    matrix = np.array([[1, importance_scale], [1 / importance_scale, 1]])
    col_sums = matrix.sum(axis=0)
    norm_matrix = matrix / col_sums
    weights = norm_matrix.sum(axis=1) / 2
    return {'w1': weights[0], 'w2': weights[1]}


def calculate_ahp_weights(criteria: list, comparison_matrix: np.ndarray) -> dict:
    """使用层次分析法 (AHP) 根据成对比较矩阵计算标准权重。"""
    n = len(criteria)
    if comparison_matrix.shape != (n, n):
        raise ValueError("比较矩阵的维度必须与标准的数量相匹配。")
    # 计算每一行的几何平均值
    geometric_means = [np.prod(row) ** (1 / n) for row in comparison_matrix]
    # 归一化几何平均值，得到权重
    total_geometric_mean = sum(geometric_means)
    weights = [gm / total_geometric_mean for gm in geometric_means]
    # 一致性检验
    lambda_max = np.max(np.linalg.eigvals(comparison_matrix)).real
    consistency_index = (lambda_max - n) / (n - 1)
    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    if n > 2:
        consistency_ratio = consistency_index / random_index.get(n, 1.45)
        print(f"AHP 一致性比率 (CR): {consistency_ratio:.4f}")
        if consistency_ratio > 0.1:
            print("⚠️ 警告: 您的成对比较判断存在不一致性 (CR > 0.1)，建议重新评估。")
    return dict(zip(criteria, weights))


# ========== 动态分类函数 ==========
def classify_attractions(df):
    classification_rules = {
        '文博艺术': {'博物馆/展馆': ['博物馆', '展馆', '展览']},
        '人文史迹': {'历史遗迹': ['历史建筑', '古镇古村']},
        '自然风光': {
            '动植物园/公园': ['公园&植物园', '植物园', '园林花园', '亲近动物'],
            '自然景观': ['自然山水', '登高爬山', '岛屿漫步', '海滩&岛屿', '海滨玩水']
        },
        '休闲娱乐': {
            '主题乐园': ['游乐场', '主题乐园'],
            '城市观光': ['地标观景', '夜游观景', '城市漫步', '乘船游览'],
            '表演/度假': ['演出', '度假村']
        },
        '综合/其他': {'综合活动': ['遛娃宝藏地', '避暑纳凉']}
    }
    tag_to_category = {}
    for l1, l2_dict in classification_rules.items():
        for l2, tags in l2_dict.items():
            for tag in tags:
                tag_to_category[tag] = (l1, l2)
    l1_col, l2_col = [], []
    for tags_str in df['标签']:
        try:
            tags_list = ast.literal_eval(str(tags_str))
            if not isinstance(tags_list, list): tags_list = []
        except (ValueError, SyntaxError):
            tags_list = []
        found_category = None
        for tag in tags_list:
            if tag in tag_to_category:
                found_category = tag_to_category[tag]
                break
        if found_category:
            l1_col.append(found_category[0])
            l2_col.append(found_category[1])
        else:
            l1_col.append('综合/其他')
            l2_col.append('其他')
    df['一级分类 (L1_Category)'] = l1_col
    df['二级分类 (L2_Category)'] = l2_col
    return df


# ========== 多样性计算函数 (使用AHP权重) ==========
def calculate_route_diversity(route_indices, attractions_df, total_l1, total_l2, ahp_weights):
    if not route_indices: return 0.0
    route_details = attractions_df.iloc[route_indices]
    unique_l1 = route_details['一级分类 (L1_Category)'].nunique()
    unique_l2 = route_details['二级分类 (L2_Category)'].nunique()
    l1_score = unique_l1 / total_l1 if total_l1 > 0 else 0
    l2_score = unique_l2 / total_l2 if total_l2 > 0 else 0
    return ahp_weights['w1'] * l1_score + ahp_weights['w2'] * l2_score


# ========== 修复后的高德API车行时间函数 ==========
def get_walk_time_api(origin, destination):
    """修复后的高德API调用函数，确保坐标系转换正确"""
    # 增加坐标验证
    if not all(k in origin for k in ['lon', 'lat']) or not all(k in destination for k in ['lon', 'lat']):
        print("❌ 坐标数据不完整，缺少经纬度信息")
        return 30.0, []

    # 验证原始坐标是否合理
    try:
        origin_lon, origin_lat = float(origin['lon']), float(origin['lat'])
        dest_lon, dest_lat = float(destination['lon']), float(destination['lat'])

        # 检查起点是否在河里（针对广州地区）
        if is_point_in_river(origin_lon, origin_lat):
            print(f"⚠️ 警告：起点 {origin.get('name', '')} 坐标位于河流区域，可能存在坐标错误")

        # 检查终点是否在河里
        if is_point_in_river(dest_lon, dest_lat):
            print(f"⚠️ 警告：终点 {destination.get('name', '')} 坐标位于河流区域，可能存在坐标错误")
    except:
        print("❌ 坐标格式错误，无法验证")
        return 30.0, []

    # 打印原始坐标用于调试
    print(
        f"📌 原始WGS84坐标 - 起点: ({origin_lon:.6f}, {origin_lat:.6f}), 终点: ({dest_lon:.6f}, {dest_lat:.6f})")

    # ⚠️ 关键修复：将WGS84坐标转换为GCJ02坐标，用于高德API调用
    origin_gcj_lon, origin_gcj_lat = wgs84_to_gcj02(origin_lon, origin_lat)
    dest_gcj_lon, dest_gcj_lat = wgs84_to_gcj02(dest_lon, dest_lat)

    print(f"🔄 转换后GCJ02坐标 - 起点: ({origin_gcj_lon:.6f}, {origin_gcj_lat:.6f}), 终点: ({dest_gcj_lon:.6f}, {dest_gcj_lat:.6f})")

    url = 'https://restapi.amap.com/v3/direction/driving'
    params = {
        'origin': f"{origin_gcj_lon:.6f},{origin_gcj_lat:.6f}",
        'destination': f"{dest_gcj_lon:.6f},{dest_gcj_lat:.6f}",
        'key': GAODE_API_KEY,
        'extensions': 'all',
        'strategy': 0,
        'output': 'json'
    }

    max_retries = 5
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
                                            # ⚠️ 关键修复：高德API返回的是GCJ02坐标，需要转换回WGS84供folium使用
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
                        sleep(2.0 if 'CUQPS_HAS_EXCEEDED_THE_LIMIT' in error_info else 1.0)
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
    # 更精确的直线距离估算
    lon_diff = math.radians(dest_lon - origin_lon)
    lat_diff = math.radians(dest_lat - origin_lat)
    a = math.sin(lat_diff / 2) ** 2 + math.cos(math.radians(origin_lat)) * math.cos(
        math.radians(dest_lat)) * math.sin(lon_diff / 2) ** 2
    distance_km = 6371 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
    time_cost = distance_km * 10  # 假设平均时速6km/h，转换为分钟
    return time_cost, coords


# ========== 获取时间矩阵和路径缓存（带缓存） ==========
def get_time_matrix_and_route_cache(points, N, cache_file):
    # 尝试从缓存加载
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data['time_matrix']) == N:  # 检查缓存是否匹配当前数据
                    time_matrix = cached_data['time_matrix']
                    route_dict = cached_data.get('route_dict', {})
                    print("✅ 从缓存加载时间矩阵和路径缓存")
                    print(f"📊 缓存统计: 时间矩阵 {len(time_matrix)}x{len(time_matrix)}, 路径数据 {len(route_dict)} 条")
                    return time_matrix, route_dict
                else:
                    print(
                        f"⚠️ 缓存数据维度不匹配: 期望 {N}x{N}, 实际 {len(cached_data['time_matrix'])}x{len(cached_data['time_matrix'])}")
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}")

    print("🔄 获取车行时间矩阵和路径坐标中...")
    time_matrix = [[0] * N for _ in range(N)]
    route_dict = {}

    # 计算总调用次数
    total_calls = N * (N - 1) // 2
    current_call = 0

    for i in range(N):
        for j in range(i + 1, N):
            current_call += 1
            print(f"进度: {current_call}/{total_calls} ({current_call / total_calls * 100:.1f}%)")

            time_val, coords = get_walk_time_api(points[i], points[j])
            time_matrix[i][j] = time_val
            time_matrix[j][i] = time_val

            # 只保存有效的路径数据
            if coords and len(coords) >= 2:  # 确保至少有起点和终点
                route_dict[f"{i}_{j}"] = coords
                route_dict[f"{j}_{i}"] = coords[::-1]  # 反向坐标
                print(f"  ✅ 保存路径 {i}_{j}: {len(coords)} 个坐标点")
            else:
                # 如果API没有返回有效坐标，生成直线路径
                try:
                    start_lon, start_lat = points[i]['lon'], points[i]['lat']
                    end_lon, end_lat = points[j]['lon'], points[j]['lat']
                    fallback_coords = [(start_lat, start_lon), (end_lat, end_lon)]
                    route_dict[f"{i}_{j}"] = fallback_coords
                    route_dict[f"{j}_{i}"] = fallback_coords[::-1]
                    print(f"  ⚠️ 路径 {i}_{j}: 使用直线路径 (2 个坐标点)")
                except Exception as e:
                    print(f"  ❌ 路径 {i}_{j}: 无法生成路径数据 - {e}")
                    route_dict[f"{i}_{j}"] = []
                    route_dict[f"{j}_{i}"] = []

    # 保存到缓存
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'time_matrix': time_matrix, 'route_dict': route_dict}, f)
        print("✅ 时间矩阵和路径缓存已保存")
        print(f"📊 缓存统计: 时间矩阵 {N}x{N}, 路径数据 {len(route_dict)} 条")
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

    print("✅ 时间矩阵和路径获取完成")
    return time_matrix, route_dict


# ========== 蚁群算法主函数 (使用AHP评分) ==========
def aco(time_limit, time_matrix, stay_times, satisfactions, attractions_df, diversity_ahp_weights, objective_weights):
    N = len(attractions_df)
    TOTAL_L1_CATEGORIES = attractions_df['一级分类 (L1_Category)'].nunique()
    TOTAL_L2_CATEGORIES = attractions_df['二级分类 (L2_Category)'].nunique()

    pheromone = np.ones((N, N))
    best_route, best_score, best_diversity = None, -float('inf'), 0
    best_time, best_stay = 0, 0

    print(f"🔄 开始蚁群算法优化... (时间限制: {time_limit / 60:.1f} 小时)")

    for it in range(ACO_ITER):
        if it % 10 == 0:
            print(f"迭代进度: {it + 1}/{ACO_ITER}")

        all_routes, all_scores = [], []
        for ant in range(ANT_COUNT):
            visited = [0]
            total_travel_time, total_stay_time = 0, 0

            while True:
                current = visited[-1]
                candidates = [i for i in range(1, N) if i not in visited]
                if not candidates: break

                probs = []
                for j in candidates:
                    eta = (satisfactions[j] + 1e-6) / ((time_matrix[current][j] + stay_times[j] + 1e-6))
                    tau = pheromone[current][j]
                    probs.append((tau ** ALPHA) * (eta ** BETA))

                probs = np.array(probs)
                if probs.sum() <= 0: break
                probs /= probs.sum()

                try:
                    next_node = np.random.choice(candidates, p=probs)
                except ValueError:
                    break

                # 预估增加新节点后的总时间 (旅行+停留+返回)
                new_travel_time = total_travel_time + time_matrix[current][next_node]
                new_stay_time = total_stay_time + stay_times[next_node]
                return_travel_time = time_matrix[next_node][0]

                if new_travel_time + new_stay_time + return_travel_time > time_limit:
                    break

                visited.append(next_node)
                total_travel_time = new_travel_time
                total_stay_time = new_stay_time

            # 最终路线和时间
            final_route = visited + [0]
            final_travel_time = total_travel_time + time_matrix[visited[-1]][0]
            final_total_time = final_travel_time + total_stay_time

            # --- 使用AHP权重进行评分 ---
            tour_indices = visited[1:]
            cover = len(tour_indices)
            total_satisfaction = sum(satisfactions[i] for i in tour_indices)
            stay_ratio = total_stay_time / (final_total_time + 1e-6)
            diversity_score = calculate_route_diversity(tour_indices, attractions_df, TOTAL_L1_CATEGORIES,
                                                        TOTAL_L2_CATEGORIES, diversity_ahp_weights)

            # 组合得分
            score = (cover * objective_weights['cover']) + \
                    (total_satisfaction * objective_weights['satisfaction']) + \
                    (stay_ratio * objective_weights['stay_ratio']) + \
                    (diversity_score * objective_weights['diversity'])

            score *= BASE_MULTIPLIER
            score -= (final_total_time * 0.01)  # 时间惩罚项

            all_routes.append(final_route)
            all_scores.append(score)

            if score > best_score:
                best_route, best_score, best_time, best_stay, best_diversity = final_route, score, final_total_time, total_stay_time, diversity_score

        # 信息素更新
        pheromone *= (1 - RHO)
        pheromone[pheromone < 0.01] = 0.01
        for idx, route in enumerate(all_routes):
            if not route or len(route) <= 2: continue
            update_score = max(0, all_scores[idx])
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i + 1]] += Q * update_score / (len(route) - 2 + 1e-6)

    return best_route, best_score, best_time, best_stay, best_diversity


# ========== 验证缓存数据完整性 ==========
def verify_cache_integrity(route_cache, N):
    """验证缓存数据的完整性"""
    print(f"🔍 验证缓存数据完整性...")
    missing_keys = []
    empty_keys = []

    for i in range(N):
        for j in range(N):
            if i != j:
                key = f"{i}_{j}"
                if key not in route_cache:
                    missing_keys.append(key)
                elif not route_cache[key]:
                    empty_keys.append(key)

    print(f"📊 缓存验证结果:")
    print(f"  - 总路径数: {N * (N - 1)}")
    print(f"  - 缺失键: {len(missing_keys)}")
    print(f"  - 空数据键: {len(empty_keys)}")

    if missing_keys:
        print(f"  ⚠️ 缺失的键: {missing_keys[:5]}...")
    if empty_keys:
        print(f"  ⚠️ 空数据的键: {empty_keys[:5]}...")

    return len(missing_keys) == 0 and len(empty_keys) == 0


# ========== 修复后的路线可视化函数 ==========
def plot_routes(df, routes, route_cache, output_html_path):
    """修复后的可视化函数，确保景点坐标正确显示"""

    print("🗺️ 开始生成地图可视化...")
    print("⚠️ 重要：景点坐标将直接使用原始WGS84坐标，不进行任何转换")

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
        tiles='OpenStreetMap',  # WGS84坐标系底图
        attr='OpenStreetMap (WGS84 坐标系)'
    )

    # 添加河流区域标记（用于调试，可选）
    folium.Polygon(
        locations=[
            [23.10, 113.25], [23.10, 113.35],
            [22.45, 113.35], [22.45, 113.25]
        ],
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.1,
        popup='珠江区域（调试用）'
    ).add_to(m)

    # ⚠️ 关键修复：添加所有景点标记，使用原始WGS84坐标
    attraction_markers = {}
    for idx, row in df.iterrows():
        try:
            # 直接使用原始WGS84坐标，不进行任何转换
            lat, lon = float(row['lat']), float(row['lon'])
            name = row['name']
            attraction_markers[idx] = (lat, lon)

            # 检查是否在河流区域
            in_river = is_point_in_river(lon, lat)

            # 添加景点标记（如果在河里则用特殊颜色警告）
            marker_color = 'red' if in_river else 'darkblue'
            popup_text = f"{name}<br>WGS84坐标: ({lat:.6f}, {lon:.6f})"
            if in_river:
                popup_text += "<br><b>⚠️ 警告：可能位于河流区域，坐标可能错误</b>"

            folium.CircleMarker(
                location=[lat, lon],  # 直接使用WGS84坐标
                radius=10,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(m)

            # 添加文字标签
            folium.map.Marker(
                [lat + 0.0005, lon],  # 微调标签位置，避免遮挡
                icon=folium.DivIcon(
                    html=f'<div style="font-size:12px; color:{marker_color}; font-weight:bold;">{name}</div>'
                )
            ).add_to(m)

            print(f"✅ 添加景点标记: {name} at WGS84 ({lat:.6f}, {lon:.6f})")

        except Exception as e:
            print(f"⚠️ 添加景点标记失败: {row.get('name', '未知')}, 错误: {e}")

    # 绘制路线
    for route_name, points in routes.items():
        print(f"\n🗺️ 绘制 {route_name} 路线...")
        complete_route = []

        for i in range(len(points) - 1):
            # 获取当前点和下一个点的信息
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
                # 直接连接这两个点
                segment_coords = [(current_lat, current_lon), (next_lat, next_lon)]
            else:
                # 使用缓存的路径（应该已经是WGS84坐标）
                key = f"{current_idx}_{next_idx}"
                segment_coords = route_cache.get(key, [])

                # 验证路径是否连接正确的景点，使用更严格的阈值
                if segment_coords:
                    start_diff = math.hypot(segment_coords[0][1] - current_lon, segment_coords[0][0] - current_lat)
                    end_diff = math.hypot(segment_coords[-1][1] - next_lon, segment_coords[-1][0] - next_lat)

                    # 如果路径起点不匹配，添加连接线
                    if start_diff > 0.00005:  # 约5米差异
                        segment_coords.insert(0, (current_lat, current_lon))
                        print(f"⚠️ 修正路径起点: {current_name} (差异: {start_diff:.7f})")

                    # 如果路径终点不匹配，添加连接线
                    if end_diff > 0.00005:  # 约5米差异
                        segment_coords.append((next_lat, next_lon))
                        print(f"⚠️ 修正路径终点: {next_name} (差异: {end_diff:.7f})")
                else:
                    # 生成直线路径
                    segment_coords = [(current_lat, current_lon), (next_lat, next_lon)]
                    print(f"⚠️ 生成直线路径: {current_name} -> {next_name}")

            # 添加到完整路线
            if complete_route and complete_route[-1] == segment_coords[0]:
                complete_route.extend(segment_coords[1:])
            else:
                complete_route.extend(segment_coords)

        # 绘制完整路线
        if complete_route:
            folium.PolyLine(
                complete_route,
                color=color_dict.get(route_name, 'gray'),
                weight=6,
                opacity=0.9,
                popup=f"{route_name}路线"
            ).add_to(m)
            print(f"✅ {route_name} 路线绘制完成，{len(complete_route)}个坐标点")

    # 添加图例
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed; 
        bottom: 30px; right: 30px; width: 240px; 
        z-index:9999; font-size:14px;
        background-color: white;
        border:2px solid grey;
        padding: 10px;
    ">
    <b>图例 (已修复WGS84坐标系)</b><br>
    <span style="color:darkblue;">●</span> 景点位置 (WGS84)<br>
    <span style="color:red;">●</span> 可能位置错误的景点<br>
    <span style="color:blue; opacity:0.3;">■</span> 河流区域（调试）<br>
    {'<br>'.join([f'<span style="color:{color_dict.get(lbl, "gray")};">━━━</span> {lbl}路线' for lbl in routes.keys()])}
    </div>
    {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)

    m.save(output_html_path)
    print(f"✅ 地图已保存为 {output_html_path}")


# ========== 缓存清理函数 ==========
def clear_cache_files(prefix="time_matrix_cache"):
    """删除所有以指定前缀开头的缓存文件"""
    cache_files = glob.glob(f"{prefix}*.pkl")

    if not cache_files:
        print("没有找到缓存文件")
        return

    for file in cache_files:
        try:
            os.remove(file)
            print(f"已删除缓存文件: {file}")
        except Exception as e:
            print(f"删除缓存文件 {file} 失败: {e}")


# ========== 主程序入口 ==========
if __name__ == '__main__':
    print("🔧 已修复景点坐标显示问题！")
    print("✅ 关键修复:")
    print("   1. 景点标记：直接使用原始WGS84坐标，不进行转换")
    print("   2. API调用：WGS84 → GCJ02 → WGS84 转换流程")
    print("   3. 底图选择：使用OpenStreetMap (WGS84坐标系)")
    print("   4. 路径坐标：确保转换回WGS84用于显示")
    print("=" * 60)
    
    # 强制清理缓存并重新生成所有路径（解决缓存导致的坐标不一致问题）
    CLEAR_CACHE_FIRST = True  # 确保获取最新路径数据
    FORCE_REFRESH_ROUTES = True  # 强制刷新路线数据

    if CLEAR_CACHE_FIRST:
        print("=" * 60)
        print("🧹 正在清理所有缓存文件...")
        clear_cache_files(CACHE_FILE_PREFIX)
        print("✅ 缓存清理完成")
        print("=" * 60 + "\n")

    # --- 步骤1: (全局) 计算AHP权重 ---
    print("=" * 60)
    print("⚖️  使用AHP计算全局权重...")
    # 1.1 多样性内部权重
    diversity_weights = calculate_ahp_2x2_weights(L1_TO_L2_IMPORTANCE)
    print(
        f"多样性权重 (L1:L2 Importance={L1_TO_L2_IMPORTANCE}) -> w1(L1): {diversity_weights['w1']:.3f}, w2(L2): {diversity_weights['w2']:.3f}")
    # 1.2 总目标函数权重
    objective_weights = calculate_ahp_weights(OBJECTIVE_CRITERIA, OBJECTIVE_COMPARISON_MATRIX)
    print("最终目标权重为:")
    for key, value in objective_weights.items():
        print(f"   - {key}: {value:.4f}")
    print("=" * 60)

    # --- 步骤2: 循环处理每个Excel文件 ---
    for i in range(1, 20):
        EXCEL_FILE = f"酒店{i}.xlsx"
        CACHE_FILE = f"{CACHE_FILE_PREFIX}_{i}.pkl"

        print(f"\n{'=' * 60}\n📄 正在处理文件: {EXCEL_FILE}\n{'=' * 60}")

        if not os.path.exists(EXCEL_FILE):
            print(f"❌ 文件不存在: {EXCEL_FILE}, 跳过...")
            continue
        try:
            df = pd.read_excel(EXCEL_FILE)
        except Exception as e:
            print(f"❌ 读取失败: {e}, 跳过...")
            continue

        # 验证并修正坐标有效性（新增自动修正功能）
        print("🔍 开始验证并修正景点坐标...")
        invalid_indices = validate_and_correct_coordinates(df)
        if invalid_indices:
            print(f"⚠️ 发现 {len(invalid_indices)} 个无效坐标，建议检查并修正原始数据")

        df = classify_attractions(df)
        df.fillna(0, inplace=True)

        # 计算满意度 (信息熵法)
        data = df[['评分', '评论数', '热度']].copy()
        data['评论数'] = np.log1p(data['评论数'].astype(float))
        norm_data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
        P = norm_data / norm_data.sum()
        E = (-1 / np.log(len(norm_data))) * (P * np.log(P + 1e-8)).sum()
        d = 1 - E
        w = d / d.sum()
        df['满意度'] = norm_data.dot(w.values)

        points = df.to_dict('records')
        N = len(points)

        # 如果需要强制刷新，删除缓存文件
        if FORCE_REFRESH_ROUTES and os.path.exists(CACHE_FILE):
            try:
                os.remove(CACHE_FILE)
                print(f"🗑️ 已删除旧缓存文件: {CACHE_FILE}，将重新生成路径")
            except Exception as e:
                print(f"⚠️ 删除缓存文件失败: {e}")

        # 获取时间矩阵和路径缓存
        time_matrix, route_cache = get_time_matrix_and_route_cache(points, N, CACHE_FILE)

        # 验证路径完整性
        verify_cache_integrity(route_cache, N)

        stay_times = df['stay'].tolist()
        satisfactions = df['满意度'].tolist()

        hotel_name = str(df.iloc[0]['name'])
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', hotel_name)
        output_filename = f"{safe_name}{OUTPUT_ROUTE_FILE_SUFFIX}"

        all_route_rows = []
        routes_for_plot = defaultdict(list)

        for limit in TIME_LIMITS:
            label = f"{int(limit / 60)}h"
            print(f"\n{'=' * 50}\n⏰ 时间上限: {limit} 分钟")

            # 调用ACO，传入计算好的两组AHP权重
            route_indices, score, total_time, total_stay, diversity_score = aco(
                time_limit=limit,
                time_matrix=time_matrix,
                stay_times=stay_times,
                satisfactions=satisfactions,
                attractions_df=df,
                diversity_ahp_weights=diversity_weights,
                objective_weights=objective_weights
            )

            if not route_indices:
                print("在此时间限制下未能规划出有效路径。")
                continue

            names = [df.iloc[i]['name'] for i in route_indices]
            tour_indices = route_indices[1:-1]

            travel_time = sum(
                time_matrix[route_indices[i]][route_indices[i + 1]] for i in range(len(route_indices) - 1))
            actual_stay = sum(stay_times[i] for i in tour_indices)
            total_satisfaction = sum(satisfactions[i] for i in tour_indices)

            print("\n--- 最终规划结果 ---")
            print("🗺️  最优路径: ", " -> ".join(names))
            print(f"🏆 路线综合得分: {score:.2f}")
            print(f"🕒 总耗时(含游览): {total_time:.1f} 分钟")
            print(f"🚗 途中耗时(车行): {travel_time:.1f} 分钟")
            print(f"🎉 实际游览时间: {actual_stay:.1f} 分钟")
            print(f"📊 游览时间占比: {actual_stay / (total_time + 1e-6):.2%}")
            print(f"⭐ 满意度累计: {total_satisfaction:.3f}")
            print(f"🎨 路线多样性得分: {diversity_score:.4f}")
            print(f"📍 覆盖景点数: {len(tour_indices)}")

            # 验证时间计算
            calculated_total = travel_time + actual_stay
            if abs(calculated_total - total_time) > 1:
                print(f"⚠️ 时间计算不一致! 算法返回: {total_time:.1f}, 重新计算: {calculated_total:.1f}")

            # 准备输出到Excel
            for order, idx in enumerate(route_indices):
                all_route_rows.append({
                    'route': label,
                    'order': order + 1,
                    'name': df.iloc[idx]['name'],
                    'lon': df.iloc[idx]['lon'],
                    'lat': df.iloc[idx]['lat'],
                    '是否在河流区域': '是' if is_point_in_river(float(df.iloc[idx]['lon']),
                                                                float(df.iloc[idx]['lat'])) else '否'
                })
                routes_for_plot[label].append(
                    (order + 1, df.iloc[idx]['name'], (df.iloc[idx]['lon'], df.iloc[idx]['lat'])))

            # 立即生成当前时间限制的路线可视化
            current_routes = {label: routes_for_plot[label]}
            current_html = f"{safe_name}_{label}_routes_map_FIXED.html"
            print(f"\n🗺️ 正在生成 {label} 路线可视化...")
            plot_routes(df, current_routes, route_cache, current_html)

        # 保存Excel最优路径
        if all_route_rows:
            pd.DataFrame(all_route_rows).to_excel(output_filename, index=False)
            print(f"\n✅ 已保存规划结果: {output_filename}")

            # 生成综合地图可视化（包含所有时间限制）
            output_html = f"{safe_name}_all_routes_map_FIXED.html"
            print(f"\n🗺️ 正在生成综合路线可视化...")
            plot_routes(df, routes_for_plot, route_cache, output_html)
        else:
            print(f"\n⚠️ 未生成任何有效路径，不保存文件。")