import math

# ========== 坐标系转换函数 (WGS84 <-> GCJ02) ==========
PI = math.pi
AXIS = 6378245.0
OFFSET = 0.00669342162296594323  # (a^2 - b^2) / a^2


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


def wgs84_to_gcj02(lon, lat):
    """将WGS84坐标转换为GCJ02坐标（火星坐标系）"""
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
    """将GCJ02坐标转换为WGS84坐标"""
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


def test_coordinate_conversion():
    """测试坐标系转换的正确性"""
    print("🧪 测试坐标系转换...")
    print("=" * 60)
    
    # 测试几个广州的真实坐标（WGS84）
    test_points = [
        ("广州塔", 113.3191, 23.1098),
        ("天河城", 113.3308, 23.1292),
        ("白云山", 113.3034, 23.1724),
        ("珠江新城", 113.3195, 23.1203)
    ]
    
    for name, wgs84_lon, wgs84_lat in test_points:
        print(f"\n📍 测试点: {name}")
        print(f"原始WGS84坐标: ({wgs84_lon:.6f}, {wgs84_lat:.6f})")
        
        # 转换为GCJ02
        gcj02_lon, gcj02_lat = wgs84_to_gcj02(wgs84_lon, wgs84_lat)
        print(f"转换后GCJ02坐标: ({gcj02_lon:.6f}, {gcj02_lat:.6f})")
        
        # 转换偏移量
        offset_lon = gcj02_lon - wgs84_lon
        offset_lat = gcj02_lat - wgs84_lat
        print(f"转换偏移量: 经度 {offset_lon:.6f}, 纬度 {offset_lat:.6f}")
        
        # 再转换回WGS84
        converted_back_lon, converted_back_lat = gcj02_to_wgs84(gcj02_lon, gcj02_lat)
        print(f"转换回WGS84坐标: ({converted_back_lon:.6f}, {converted_back_lat:.6f})")
        
        # 计算误差
        error_lon = abs(converted_back_lon - wgs84_lon)
        error_lat = abs(converted_back_lat - wgs84_lat)
        print(f"转换误差: 经度 {error_lon:.8f}, 纬度 {error_lat:.8f}")
        
        if error_lon < 1e-6 and error_lat < 1e-6:
            print("✅ 坐标系转换正确")
        else:
            print("❌ 坐标系转换存在误差")
    
    print("\n" + "=" * 60)
    print("📋 问题分析总结:")
    print("1. WGS84 → GCJ02: 用于调用高德API")
    print("2. GCJ02 → WGS84: 将API返回的路径转换回来用于folium显示")
    print("3. 景点坐标显示: 直接使用原始WGS84坐标，不进行转换")
    print("4. folium底图: 使用OpenStreetMap (WGS84坐标系)")


def analyze_coordinate_issue():
    """分析您代码中的坐标问题"""
    print("\n🔍 分析您代码中的坐标问题:")
    print("=" * 60)
    
    print("❌ 问题1: 在可视化函数plot_routes()中，可能对景点坐标进行了错误的转换")
    print("   解决方案: 景点标记应该直接使用原始WGS84坐标，不要转换")
    
    print("\n❌ 问题2: 路径坐标和景点坐标可能使用了不同的坐标系")
    print("   解决方案: 确保所有用于folium显示的坐标都是WGS84格式")
    
    print("\n❌ 问题3: 可能使用了错误的底图坐标系")
    print("   解决方案: 使用OpenStreetMap底图，它使用WGS84坐标系")
    
    print("\n✅ 正确的流程应该是:")
    print("1. 原始数据: WGS84坐标")
    print("2. 调用高德API: WGS84 → GCJ02")
    print("3. API返回路径: GCJ02 → WGS84 (用于folium)")
    print("4. 景点显示: 直接使用原始WGS84坐标")
    print("5. folium底图: OpenStreetMap (WGS84)")


if __name__ == '__main__':
    test_coordinate_conversion()
    analyze_coordinate_issue()