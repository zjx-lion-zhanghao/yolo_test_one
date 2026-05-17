import os
import xml.etree.ElementTree as ET
class_count = {}
folder_path = r'C:\Users\78456\Desktop\data\Annotations'  # 此处修改为自己的xml文件夹路径
for filename in os.listdir(folder_path):
	if filename.endswith('.xml'):
		tree = ET.parse(os.path.join(folder_path, filename))
		root = tree.getroot()
		for obj in root.findall('object'):
			name = obj.find('name').text
			if name in class_count:
				class_count[name] += 1
			else:
				class_count[name] = 1
sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
print(sorted_class_count)
print("各类别数量（从大到小）:")
for name, count in sorted_class_count:
	print(f"{name}: {count}")