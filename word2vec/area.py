from pyproj import Proj

# UTM投影参数（区号49N）
utm_proj = Proj(proj='utm', zone=49, ellps='WGS84', south=False)

# 输入点（经纬度格式：[纬度, 经度]）
points = {
    "光明联想创新科技园": [22.7938, 113.9408],
    "龙岗低碳城": [22.7847, 114.2995],
    "浪骑游艇会": [22.5594, 114.5404],
    "蛇口邮轮母港": [22.4746, 113.9197],
    "宝安机场": [22.6375, 113.8184]
}

# 转换为UTM坐标（东向Easting, 北向Northing）
utm_coords = {}
for name, coord in points.items():
    lat, lon = coord[0], coord[1]
    easting, northing = utm_proj(lon, lat)
    utm_coords[name] = (easting, northing)

# 输出UTM坐标（示例值，实际值需运行代码获取）
print(utm_coords)

# 按顺序排列的点（假设为闭合五边形）
ordered_names = ["光明联想创新科技园", "龙岗低碳城", "浪骑游艇会", "蛇口邮轮母港", "宝安机场"]
ordered_coords = [utm_coords[name] for name in ordered_names]

# Shoelace公式计算面积
def polygon_area(coords):
    n = len(coords)
    area = 0.0
    for i in range(n):
        xi, yi = coords[i]
        xj, yj = coords[(i+1) % n]
        area += (xi * yj - xj * yi)
    return abs(area) / 2.0

# 计算面积（平方米）
area_m2 = polygon_area(ordered_coords)
# 转换为平方公里
area_km2 = area_m2 / 1e6
print(f"五边形面积: {area_km2:.2f} 平方公里")