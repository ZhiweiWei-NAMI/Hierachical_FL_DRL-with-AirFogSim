import bs4
import xlsxwriter
 
# 读取xml文件，写入excel
def xmlToExcel(file_xml, file_excel):
    # 打开xml文件，并以此创建一个bs对象
    xml = open(file_xml, 'r')
    doc = bs4.BeautifulSoup(xml, 'xml')
 
    # 创建一个excel文件，并添加一个sheet，命名为orders
    workbook = xlsxwriter.Workbook(file_excel)
    sheet = workbook.add_worksheet('orders')
 
    # 设置粗体
    bold = workbook.add_format({'bold': True})
 
    # 先在第一行写标题，用粗体
    sheet.write('A1', u'time', bold)
    sheet.write('B1', u'id', bold)
    sheet.write('C1', u'x', bold)
    sheet.write('D1', u'y', bold)
    sheet.write('E1', u'angle', bold)
    sheet.write('F1', u'speed', bold)
    sheet.write('G1', u'lane', bold)
    sheet.write('H1', u'pos', bold)
    #从第二行开始录入数据
    row = 2
    # 筛选出所有的<timestep>，这里使用的是CSS选择器
    times = doc.select('timestep')
    for t in times:
        print("--------------------")
        print(t)
        #获取时间戳
        time = t.attrs["time"]
        #获取车辆中数据
        vehicle = t.select('vehicle')
        print("%%%%%%%%%%%%%%%%%%%%%")
 
        for v in vehicle:
            id = v.attrs['id']
            x_num = v.attrs['x']
            y_num = v.attrs['y']
            angle = v.attrs['angle']
            speed = v.attrs['speed']
            lane = v.attrs['lane']
            pos = v.attrs['pos']
 
            print("("+id+","+x_num+","+y_num+","+angle+","+speed+","+lane+","+pos+")")
            # 将每个timestep中车辆数据写入excel中
            sheet.write('A%d' % row, time)
            sheet.write('B%d' % row, id)
            sheet.write('C%d' % row, x_num)
            sheet.write('D%d' % row, y_num)
            sheet.write('E%d' % row, angle)
            sheet.write('F%d' % row, speed)
            sheet.write('G%d' % row, lane)
            sheet.write('H%d' % row, pos)
 
            row += 1
    # 关闭文件
    xml.close()
    workbook.close()
 
# 测试代码
if __name__ == '__main__':
    file1 = r'E:\scholar\papers\uav_compute\sumo\sumoTrace.xml'
    file2 = r'E:\scholar\papers\uav_compute\sumo\sumoTrace.xlsx'
 
    xmlToExcel(file1, file2)