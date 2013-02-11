# coding=utf-8
import Image
import numpy
from shutil import copyfile
import os
from wx import PyDeadObjectError
from itertools import chain
from collections import Counter
from math import *
from time import localtime, strftime
#TODO:check imports

def whitePixelGenerator(matrix):
    for y in xrange(1, matrix.shape[0] - 1):
        for x in xrange(1,matrix.shape[1]-1):
            if matrix[y,x]:
                yield x, y

def whitePixelNeGenerator(matrix):
    asdf = []
    for y in range(1, matrix.shape[0] - 1):
        for x in range(1, matrix.shape[1] - 1):
            if not int(matrix[y, x]) == 0:
                asdf.append((x, y))
    return asdf

def open_image(path):
    try:
        raster_image = Image.open(path)
        valid_raster = True
    except IOError:
        valid_raster = False
    if valid_raster:
        raster_path = "10100101.%s" % path.split('.')[-1]
        copyfile(path, raster_path)
        os.rename(raster_path, '10100101')
    return valid_raster

def delete_files():
    files = os.listdir('.')
    for i in files:
        if i == '10100101' or i == '165.svg':
            os.remove(i)

def save(path):
    try:
        copyfile('165', path)
        return True
    except IOError:
        return False
    except UnicodeWarning:
        pass

def updateStatus(window, text):
    try:
        window.SetStatusText(text)
        return True
    except PyDeadObjectError:
        return False

def convolve(arr1, arr2):
    arr = arr1*arr2
    return numpy.sum(arr)

def approximate_angle(angle):
    etalons = numpy.array([0, 45, 90, 135, 180])
    angles = numpy.array([angle]*5)
    distances = etalons - angles
    distances = numpy.absolute(distances)
    index = distances.argmin()
    etalons[4] = 0
    return etalons[index]

def gauss3(pixels, window):
    YX = pixels.shape
    blur = numpy.zeros_like(pixels, dtype=int)
    for y in xrange(1,YX[0]-1):
        if not updateStatus(window, 'Gauss...%s%%' % str(int(float(y)*100/YX[0]))):
            return
        for x in xrange(1,YX[1]-1):
            blur[y,x] = convolve(
                numpy.array([
                    (1,2,1),
                    (2,4,2),
                    (1,2,1)
                ]),
                pixels[y-1:y+2,x-1:x+2]
            )/16
    #TODO:delete
    Image.fromarray(blur).convert('L').save('2.bmp','BMP')
    return blur

def gauss5(pixels, window):
    YX = pixels.shape
    blur = numpy.zeros_like(pixels, dtype=int)
    for y in xrange(2,YX[0]-2):
        if not updateStatus(window, 'Gauss...%s%%' % str(int(float(y)*100/YX[0]))):
            return
        for x in xrange(2,YX[1]-2):
            blur[y,x] = convolve(
                numpy.array([
                    (2,7,12,7,2),
                    (7,31,52,31,7),
                    (15,52,127,52,15),
                    (7,31,52,31,7),
                    (2,7,12,7,2)
                ]),
                pixels[y-2:y+3,x-2:x+3]
            )/423
    #TODO: delete next line
    Image.fromarray(blur).convert('L').save('5.bmp','bmp')
    return blur

def preprocessing(pixels, threshold, window):
    YX = pixels.shape
    blur = numpy.zeros_like(pixels, dtype=int)
    for y in xrange(1,YX[0]-1):
        for x in xrange(1,YX[1]-1):
            if not updateStatus(window, 'Gradient approximation...%s%%' % str(int(float(y) * 100 / YX[0]))):
                return
            if pixels[y, x] > threshold:
                blur[y,x] = convolve(
                    numpy.array([
                        (1,1,1),
                        (1,0,1),
                        (1,1,1)
                    ]),
                    pixels[y-1:y+2,x-1:x+2]
                )/3
    #TODO:delete
    Image.fromarray(numpy.uint8(blur)).save('4.bmp','BMP')
    return blur

def sobel(pixels, window):
    YX = pixels.shape
    G_x = numpy.zeros_like(pixels, dtype=int)
    G_y = numpy.zeros_like(pixels, dtype=int)
    G = numpy.zeros_like(pixels, dtype=int)
    Theta = numpy.zeros_like(pixels, dtype=int)
    for y in xrange(1,YX[0]-1):
        if not updateStatus(window, 'Sobel operator...%s%%' % str(int(float(y)*100/YX[0]))):
            return
        for x in xrange(1,YX[1]-1):
            G_x[y,x] = g_x = convolve(numpy.array([(-1,0,1),(-2,0,2),(-1,0,1)]),pixels[y-1:y+2,x-1:x+2])/6
            G_y[y,x] = g_y = convolve(numpy.array([(-1,-2,-1),(0,0,0),(1,2,1)]),pixels[y-1:y+2,x-1:x+2])/6
            G[y,x] = int(hypot(g_x, g_y))
            angle = degrees(atan2(float(g_y),g_x))
            Theta[y,x] = approximate_angle(angle)
    #TODO: delete next line
    Image.fromarray(G).convert('L').save('3.bmp','BMP')
    return G, Theta

def canny(value, angle, window):
    YX = value.shape
    borders = numpy.zeros_like(value)
    for y in xrange(1,YX[0]-1):
        if not updateStatus(window, 'Canny operator...%s%%' % str(int(float(y)*100/YX[0]))):
            return
        for x in xrange(1,YX[1]-1):
            if angle[y,x] == 0:
                borders[y,x] = 255*bool(value[y,x]>value[y-1,x] and value[y,x]>value[y+1,x])
            elif angle[y,x] == 90:
                borders[y,x] = 255*bool(value[y,x]>value[y,x-1] and value[y,x]>value[y,x+1])
            elif angle[y,x] == 135:
                borders[y,x] = 255*bool(value[y,x]>value[y-1,x-1] and value[y,x]>value[y+1,x+1])
            elif angle[y,x] == 45:
                borders[y,x] = 255*bool(value[y,x]>value[y-1,x+1] and value[y,x]>value[y+1,x-1])
    #TODO: delete next line
    Image.fromarray(borders).convert('1').save('6.bmp','BMP')
    return borders

def labeling(pixels, neighbours_min_count, window):
    YX = pixels.size
    matrix = numpy.zeros_like(pixels, dtype=numpy.uint32)  # Матрица с цветами
    white_pixels = whitePixelNeGenerator(pixels)
    white_pixels_count = len(white_pixels)
    colour_conflicts = []

    #Раскраска
    current_colour = 1
    for index, pixel in enumerate(white_pixels):
        if not updateStatus(window, 'Gradient denoising...%s%%' % str(int(float(index) * 20 / white_pixels_count))):
            return
        x, y = pixel
        neighbours = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if not i == j == 0:
                    if not matrix[y + i, x + j] == 0:
                        neighbours.append(matrix[y + i, x + j])
        if neighbours:
            matrix[y, x] = min(neighbours)
            neighbours_colours = list(set(neighbours))
            if not len(neighbours_colours) == 1:
                if not neighbours_colours in colour_conflicts:
                    colour_conflicts.append(neighbours_colours)
        else:
            matrix[y, x] = current_colour
            current_colour += 1
        #print current_colour, '.', pixel, '\t', matrix[y, x], '-', neighbours
    #Image.fromarray(numpy.uint8(matrix)).save(r'C:\Users\User10100101\Desktop\step1.bmp', 'BMP')
    #print colour_conflicts

    #тут я сам не понял что написал вернее понял но чуть не запутался
    conflicted_colours = list(chain.from_iterable(colour_conflicts))
    conflicted_colours_count = len(conflicted_colours)
    colours_relation = {i : set() for i in conflicted_colours}
    #iii = 0
    for index, conflict in enumerate(colour_conflicts):
        if not updateStatus(window,
            'Gradient denoising...%s%%' % str(20 + int(float(index) * 20 / conflicted_colours_count))):
            return
        #iii+=1
        for colour in conflict:
            colours_relation[colour].update(conflict)
            set_ = set()
            for colour_ in colours_relation[colour]:
                if not colour_ == colour:
                    set_.update(colours_relation[colour])
                    set_.update(colours_relation[colour_])
                    colours_relation[colour_].update(set_)
            colours_relation[colour].update(set_)
            #print iii, 'out of', len(colour_conflicts), conflict, colours_relation[colour]
    #for item in colours_relation.items():
    #    print item[0], ':', list(item[1])

    colours_map = {i[0] : min(i[1]) for i in colours_relation.items()}
    for colour in xrange(1, current_colour + 1):
        #print 'colours_map', colour
        if not updateStatus(window,
            'Gradient denoising...%s%%' % str(40 + int(float(colour) * 20 / current_colour))):
            return
        if colour not in conflicted_colours:
            colours_map[colour] = colour

    matrix_ = numpy.zeros_like(pixels, dtype=numpy.uint32)  # Матрица с нужными цветами
    used_colours = []
    for index, pixel in enumerate(white_pixels):
        if not updateStatus(window,
            'Gradient denoising...%s%%' % str(60 + int(float(index) * 20 / white_pixels_count))):
            return
        #print 'used_colours', pixel
        x, y = pixel
        colour = colours_map[matrix[y, x]]
        used_colours.append(colour)
        matrix_[y, x] = colour
    #Image.fromarray(numpy.uint8(matrix_*10)).save(r'C:\Users\User10100101\Desktop\step2.bmp', 'BMP')#

    #print 'colours_counter start'
    colours_counter = Counter(used_colours)  # Подсчет использования цветов
    #print 'colours_counter end'
    #print colours_counter
    #Удаление пикселей с числом соседей меньшим neighbours_min_count
    pixels_ = numpy.copy(pixels)
    for index, pixel in enumerate(white_pixels):
        if not updateStatus(window,
            'Gradient denoising...%s%%' % str(80 + int(float(index) * 20 / white_pixels_count))):
            return
        x, y = pixel
        if colours_counter[matrix_[y, x]] < neighbours_min_count:
            pixels_[y, x] = 0
    Image.fromarray(numpy.uint8(pixels_)).save('5.bmp', 'BMP')
    return pixels_

def vectorize(window):
    #TODO: vectorize
    if window.chosenMethod == 1:
        houghTransform = combinatorial
    elif window.chosenMethod == 2:
        houghTransform = hierarchical
    elif window.chosenMethod == 3:
        houghTransform = adaptive
    else:
        raise IndexError("'method_id' not in [1, 2, 3]")

    if not updateStatus(window, 'Converting to grayscale...'):
        return
    pixels = numpy.asarray(Image.open('10100101').convert('L'))
    #TODO:delete
    Image.fromarray(pixels).save('1.jpg', 'JPEG')

    if not updateStatus(window, 'Gauss...'):
        return
    blur = gauss3(pixels, window)

    if not updateStatus(window, 'Sobel operator...'):
        return
    gradient_value, gradient_angle = sobel(blur, window)

    #if not updateStatus(window, 'Gauss...'):
    #    return
    #gradient_value_ = gauss5(preprocessing(gradient_value, 50), window)

    if not updateStatus(window, 'Gradient approximation...'):
        return
    gradient_value_ = preprocessing(gradient_value, 50, window)

    if not updateStatus(window, 'Gradient denoising...'):
        return
    gradient_value__ = labeling(gradient_value_, 50, window)

    if not updateStatus(window, 'Canny operator...'):
        return
    borders = canny(gradient_value__, gradient_angle, window)
    #borders = canny(preprocessing(gradient_value, 50), gradient_angle, window)

    #borders_ = qwerqwer(borders)

    if not updateStatus(window, 'Hough transform...'):
        return
    lines = houghTransform(borders, window=window)

    #f = open('trollface.svg')
    #troll_lines = f.readlines()
    #f.close()
    #f = open('165', 'w')
    #f.writelines(troll_lines)
    #f.close()

def XYtoRThetaIndexes(pixel, R, Theta, step):
    x, y = pixel
    indexes = []
    for i in xrange((len(Theta)) - 1):
        r = x*cos(radians(Theta[i])) + y*sin(radians(Theta[i]))
        if r > 0:
            for j in xrange((len(R))):
                if abs(r - R[j]) <= step[1]:
                    indexes.append((j, i))
        elif r == 0:
            indexes.append((0, i))
    return indexes

def find_R(pixel, theta, R_):
    x, y = pixel
    r = x * cos(radians(theta)) + y * sin(radians(theta))
    if r == 0:
        return 0
    elif r < 0:
        return r
    else:
        for i in xrange((len(R_)) - 1):
            if R_[i] < r <= R_[i+1]:
                return i


def point_near_line(x, y, r, theta, step):
    """
    Проверяет лежит ли точка в допустимом диапазоне r, theta.

    Args:
        x: Координата проверяемой точки.
        y: Координата проверяемой точки.
        r: Параметр центральной точки аккамулятора
        theta: Параметр центральной точки аккамулятора
        step: Размер аккамулятора
    """
    #Находим параметры некоторой прямой (2), проходящей через рассматриваемую точку
    R = hypot(x,y)
    Theta = degrees(asin(float(y)/R))

    #Находим координаты точки пересечения прямой (2) и прямой переданной в функцию (r, theta)
    x_ = (sin(radians(theta))*R - sin(radians(Theta))*r) / \
         (sin(radians(theta))*cos(radians(Theta)) - sin(radians(Theta))*cos(radians(theta)))
    y_ = - x_/(tan(radians(theta))) + R/sin(radians(theta))

    #Проверяем лежит ли пересечение рядом с рассматриваемой точкой
    R_ = hypot(x_, y_)
    Theta_ = degrees(asin(float(y_)/R_))
    near_r =  abs(R_ - R) <= step[0]#0.5 * step[0]
    near_theta = abs(Theta_ - Theta) <= step[1]#0.5 * step[1]

    return near_r and near_theta

def combinatorial(pixels, step=(2, 1), window=None):
    """
    Combinatorial Hough transform.

    Args:
        pixels: numpy 2d array of bitmap borders image
        window: main window (for updating statusbar)
        step: tuple of integers (delta R, delta Theta)

    """
    YX = pixels.shape

    #STEP 1
    #Для R
    #R ограничено размерами входного изображения
    R_max = hypot(YX[0],YX[1])
    #Границы сетки по R
    R_ = []
    i = 0
    while i <= ceil(R_max):
        R_.append(i)
        i += step[0]
    if not R_[-1] == R_max:  # Если R_max не кратно step[0]
        R_.append(R_max)
    #Центральные значения в ячейках
    R = [float(R_[i-1] + R_[i]) / 2 for i in xrange(1, len(R_))]

    #Для Theta
    #Theta в пределах [0, 2*pi]
    Theta_ = []  # Границы сетки по R
    i = 0
    while i <= 360:
        Theta_.append(i)
        i += step[1]
    if not Theta_[-1] == 360:
        Theta_.append(360)
    #Центральные значения в ячейках
    Theta = [float(Theta_[i-1] + Theta_[i]) / 2 for i in xrange(1, len(Theta_))]

    #STEP 2
    aaaaaaaaa = """    print 'step2'
    accumulator = numpy.zeros((len(R), len(Theta)), dtype=numpy.int32)
    white_pixels = whitePixelNeGenerator(pixels)
    print white_pixels
    white_pixels_count = len(white_pixels)
    for index, pixel in enumerate(white_pixels):
        if index%100==0:
            print 'step2', index, white_pixels_count
        r_theta_ = XYtoRThetaIndexes(pixel, R, Theta, step)
            #print r_theta_
        for r_theta in r_theta_:
            accumulator[r_theta[0], r_theta[1]] += 1

    #TODO: delete next line
    #print accumulator
    numpy.save('accumulator165', accumulator)"""

    accumulator = numpy.load('accumulator165 - 3 05.npy')
#    accumulator = numpy.load('accumulator165.npy')
    #STEP 3,4
    was = None
    lines_img = numpy.zeros_like(pixels)
    for i in xrange(1000):
        argmax_value = numpy.unravel_index(
            accumulator.argmax(),
            accumulator.shape
        )
        if argmax_value == was:
            pass
        else:
            was = argmax_value
            #print argmax_value
            line_img = numpy.zeros_like(pixels)
            #print 'line %s detection'%i
            #noinspection PyTypeChecker
            r1, r2, r3 = R_[argmax_value[0]], R[argmax_value[0]], R_[argmax_value[0] + 1]
            #noinspection PyTypeChecker
            theta1, theta2, theta3 = Theta_[argmax_value[1]], Theta[argmax_value[1]], Theta_[argmax_value[1] + 1]
            for x in xrange(1, pixels.shape[1] - 1):
                for y in xrange(1, pixels.shape[0] - 1):
                    for theta in (theta1, theta2, theta3):
                        if r1 <= x*cos(radians(theta)) + y*sin(radians(theta)) <= r3:
                            line_img[y, x] = 255
            #print numpy.uint8(asdf)
            max_value = accumulator.max()
            print '#%s.\t'%i + ('%s' % str(argmax_value)).ljust(12) + ('%s' % str(max_value)).ljust(4) + '%s %s' % (accumulator.std(), accumulator.mean()) + '\t%s' % strftime("%H:%M:%S", localtime())
            #print '#%s.\tmax_cell%s\t%s\t%s\t%s'%(i, argmax_value, max_value, accumulator.std(), accumulator.mean())
            #print argmax_value[0]*10 + 5, argmax_value[1]
            Image.fromarray(numpy.uint8((line_img).clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\line\%s.bmp' % i, 'BMP')
            Image.fromarray(numpy.uint8((line_img+pixels).clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\line_\%s.bmp' % i, 'BMP')
            lines_img += line_img
            Image.fromarray(numpy.uint8(lines_img.copy().clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\lines\%s.bmp' % i, 'BMP')
            Image.fromarray(numpy.uint8((lines_img+pixels).clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\lines_\%s.bmp' % i, 'BMP')
            Image.fromarray(numpy.uint8((accumulator*15).clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\acc\%s.bmp' % i, 'BMP')

            line_pixels = set(whitePixelNeGenerator(line_img))#set(whitePixelNeGenerator(pixels)) & set(whitePixelNeGenerator(line_img))
            #qwer = whitePixelNeGenerator(asdf)
            line_pixels_len = len(line_pixels)
            accumulator_ = numpy.zeros((len(R), len(Theta)), dtype=numpy.int32)
            for index, pixel in enumerate(line_pixels):
                #if index%100==0:
                    #print 'line %s deleting'%i, index, line_pixels_len
                for r_theta in XYtoRThetaIndexes(pixel, R_, Theta_, step):
                    accumulator_[r_theta[0], r_theta[1]] += 1
            #print 'aaaaaaa',accumulator.shape,accumulator_.shape
            accumulator -= accumulator_
            accumulator = accumulator.clip(0)
            Image.fromarray(numpy.uint8(accumulator.clip(0,255))).save(r'C:\Users\User10100101\Desktop\1\acc_\%s.bmp' % i, 'BMP')


def hierarchical(pixels, window):
    #TODO: hierarchical
    print 'hierarchical'

def adaptive(pixels, window):
    #TODO: adaptive
    print 'adaptive'


if __name__ == '__main__':
    img = Image.open(u'6.bmp')
    print img
    pixels = numpy.asarray(img.convert('L'), dtype=numpy.uint8)#.convert('L'))
    combinatorial(pixels, (3,0.5))