# coding:utf-8
import Image
asdf = Image.open(r'C:\Users\User10100101\Documents\Политех\5-й семестр\Компьютерная Графика\APV\borders.bmp')
for i in xrange(1,asdf.size[0]-1):
	for j in xrange(1,asdf.size[1]-1):
		if asdf.getpixel((i,j)):
			flag = False
			for k in (-1,0,1):
				for m in (-1,0,1):
					if not (k == 0 and m == 0):
						if asdf.getpixel((i+k, j+m)):
							flag = True
			if not flag:
				asdf.putpixel((i,j), 0)

				
asdf.save(r'C:\Users\User10100101\Documents\Политех\5-й семестр\Компьютерная Графика\APV\borders_.bmp', 'BMP')
