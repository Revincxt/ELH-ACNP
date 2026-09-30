import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)


# x坐标
x = np.arange(1, 1001)

# 生成y轴数据，并添加随机波动
y1 = np.log(x)
indexs = np.random.randint(0, 1000, 800)
for index in indexs:
    y1[index] += np.random.rand() - 0.5

y2 = np.log(x)
indexs = np.random.randint(0, 1000, 800)
for index in indexs:
    y2[index] += np.random.rand() - 0.5

y3 = np.log(x)
indexs = np.random.randint(0, 1000, 800)
for index in indexs:
    y3[index] += np.random.rand() - 0.5




# 绘制主图
fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(x,y1,color='#f0bc94',label='trick-1',alpha=0.7)
ax.plot(x,y2,color='#7fe2b3',label='trick-2',alpha=0.7)
ax.plot(x,y3,color='#cba0e6',label='trick-3',alpha=0.7)
ax.legend(loc='right')

# plt.show()



# 绘制缩放图
axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))

# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins.plot(x,y1,color='#f0bc94',label='trick-1',alpha=0.7)
axins.plot(x,y2,color='#7fe2b3',label='trick-2',alpha=0.7)
axins.plot(x,y3,color='#cba0e6',label='trick-3',alpha=0.7)

# 局部显示并且进行连线
zone_and_linked(ax, axins, 100, 150, x , [y1,y2,y3], 'right')

plt.show()



#
# # 绘制缩放图
# axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))
#
# # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
# axins.plot(x,y1,color='#f0bc94',label='trick-1',alpha=0.7)
# axins.plot(x,y2,color='#7fe2b3',label='trick-2',alpha=0.7)
# axins.plot(x,y3,color='#cba0e6',label='trick-3',alpha=0.7)
#
# # 局部显示并且进行连线
# zone_and_linked(ax, axins, 700, 760, x , [y1,y2,y3], 'bottom')
# plt.show()

