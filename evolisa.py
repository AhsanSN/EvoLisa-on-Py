import numpy as np
from PIL import Image as plImage
from PIL import ImageDraw as plDraw
from random import *
import math
import heapq #finding nth largest
from numpy.random import randint

# selection algorithms
# chromosome = [shapes, points, colors, image, score]
def fitnessProportionalSelection(population, nSelect):
    fitnessSum = 0
    fitnessProportions = []
    parentsIndices = []
    appendValue = 0
    for i in population:
        fitnessSum = fitnessSum + i[4]
    for i in population:
        fitnessProportions.append(i[4]/fitnessSum)
    #choosing parents
    for n in range (nSelect):
        findParentLoop = True
        if(len(parentsIndices)==len(population)):
            findParentLoop = False;
        while(findParentLoop == True):
            randNo = random()
            lower = 0
            for i in range (len(population)):
                if ((randNo>lower) and (randNo<lower + fitnessProportions[i])):
                    if(i not in parentsIndices):
                        parentsIndices.append(i)
                        findParentLoop = False
                lower = lower + fitnessProportions[i]
    return parentsIndices


def nth_largest(n, iter):
    return heapq.nlargest(n, iter)[-1]

def rankbasedSelection(population, nSelect):
    rankSum = 0
    ranksProportion = []
    parentsIndices = []
    fitness = []
    #creating array for storing ranks
    ranks = []
    for i in range (len(population)):
        ranks.append(0)
        fitness.append(population[i][4])
    #finding ranks
    for i in range (1,len(population)+1):
        nthLargestElement = nth_largest(i, fitness)
        for j in range (len(population)):
            if (nthLargestElement == population[j][4]):
                ranks[j] = i;
    # rank sum for calculation proportions
    for i in ranks:
        rankSum = rankSum + i
    for i in ranks:
        ranksProportion.append(i/rankSum)
    #choosing parents
    for n in range (nSelect):
        findParentLoop = True
        if(len(parentsIndices)==len(fitness)):
            findParentLoop = False;
        while(findParentLoop == True):
            randNo = random()
            lower = 0
            for i in range (len(fitness)):
                if ((randNo>lower) and (randNo<lower + ranksProportion[i])):
                    if(i not in parentsIndices):
                        parentsIndices.append(i)
                        findParentLoop = False
                lower = lower + ranksProportion[i]
    return parentsIndices
   
def binaryTournament(population, nSelect):
    parentsIndices = []
    pool = []
    poolLoop = True
    poolBest = 0
    #players for pool1
    for n in range (nSelect):
        while (poolLoop):
            for i in range (2):
                randNo = randint(0, len(population)-1)
                if(len(parentsIndices)==len(population)-1):
                    poolLoop = False;
                if ((randNo not in parentsIndices) and (len(parentsIndices)<len(population)+1)):
                    if(len(pool)==1):
                        if (pool[0] != randNo):
                            pool.append(randNo)
                            poolLoop = False
                    if(len(pool)==0):
                        pool.append(randNo)
        #best from pool 1            
        if (len(pool)==2):
            # for parent select
            if(population[pool[0]][4]>population[pool[1]][4]):
                poolBest = pool[0];
            if(population[pool[0]][4]<population[pool[1]][4]):
                poolBest = pool[1];
            parentsIndices.append(poolBest)
                
        if (len(parentsIndices)<nSelect):
            poolLoop = True
            pool = []
            poolBest = 0
    return(parentsIndices)

def randomSelection(population, nSelect):
    selectedIndex = []
    for i in range (nSelect):
        isChildComplete = False
        while(isChildComplete==False):
            randNo = randint(0, len(population)-1)
            if (randNo) not in selectedIndex:
                selectedIndex.append(randNo)
                isChildComplete = True;
    return selectedIndex

def truncation(population, nSelect):
    selectedIndex = []
    #finding ranks
    for i in range (1,nSelect+1):
        nthLargestElement = nth_largest(i, fitness)
        for j in range (len(fitness)):
            if (nthLargestElement == population[j][4]):
                selectedIndex.append(j)
                break
    return selectedIndex

# crossover [shapes, points, colors, image, score]
def crossOver(parentsIndex, population, n_shapes, width, height, internal):
    startCopyIndex = randint(1, n_shapes-(5)) 
    finishCopyIndex = randint(startCopyIndex, n_shapes-1)  
    parent1 = population[parentsIndex[0]]
    parent2 = population[parentsIndex[1]]
    child1 = [[],[],[],[],[]]
    child2 = [[],[],[],[],[]]
    #generating template
    for i in range (3):
        for j in range(n_shapes):
            child1[i].append(-222)
            child2[i].append(-222)
    #generating child
    #print(startCopyIndex, finishCopyIndex)
  
    # child 1
    child1[0][startCopyIndex:finishCopyIndex] = parent1[0][startCopyIndex:finishCopyIndex] #shape
    child1[1][startCopyIndex:finishCopyIndex] = parent1[1][startCopyIndex:finishCopyIndex] #point
    child1[2][startCopyIndex:finishCopyIndex] = parent1[2][startCopyIndex:finishCopyIndex] #color
    isChildComplete = False;
    childIndex = finishCopyIndex
    ParentIndex = finishCopyIndex
    #print("parent",parent1)
    #print("child",child1)
    
    while(isChildComplete==False):
        child1[0][childIndex] = parent2[0][ParentIndex]; #shape
        child1[1][childIndex] = parent2[1][ParentIndex]; #point
        child1[2][childIndex] = parent2[2][ParentIndex]; #color
        childIndex+= 1
        ParentIndex+= 1
        if (ParentIndex==len(child1[1])):
            ParentIndex = 0
            childIndex = 0
        if(childIndex == startCopyIndex):
            isChildComplete = True
    # child 2
    #copying shape point and color
    child2[0][startCopyIndex:finishCopyIndex] = parent2[0][startCopyIndex:finishCopyIndex]
    child2[1][startCopyIndex:finishCopyIndex] = parent2[1][startCopyIndex:finishCopyIndex]
    child2[2][startCopyIndex:finishCopyIndex] = parent2[2][startCopyIndex:finishCopyIndex]
    isChildComplete = False;
    childIndex = finishCopyIndex
    ParentIndex = finishCopyIndex
    while(isChildComplete==False):
        child2[0][childIndex] = parent1[0][ParentIndex];
        child2[1][childIndex] = parent1[1][ParentIndex];
        child2[2][childIndex] = parent1[2][ParentIndex];
        childIndex+= 1
        ParentIndex+= 1
        if (ParentIndex==len(child2[1])):
            ParentIndex = 0
            childIndex = 0
        if(childIndex == startCopyIndex):
            isChildComplete = True

    '''
    #[shapes, points, colors, image, score]
    child1[3] = np.array(draw_image(width, height, child1[0], child1[1], child1[2]))
    child1[4] = error_abs(internal, child1[3])
    child2[3] = np.array(draw_image(width, height, child2[0], child2[1], child2[2]))
    child2[4] = error_abs(internal, child2[3])
    '''
    return([child1, child2])

#supporting functions
def resizer(original, internal_size):
    factor = max(original.size) / internal_size
    '''
    factor = height or width per unit specified resolution 
    '''
    def reduce(image):
        '''Reduces source image to internal resolution'''
        reduced, w, h = image.copy(), original.size[0] / factor, original.size[1] / factor
        reduced.thumbnail((w, h))
        return np.array(reduced), reduced.size[0], reduced.size[1]
    
    def restore(shapes, points, colors):
        '''Restores generated image to original resolution'''
        return draw_image(*original.size, shapes * factor, points, colors, antialiasing=True)
    
    return reduce, restore

def initialize(n_shapes, min_points, max_points, width, height):
    '''Initializes random polygons for target image'''
    shapes = np.empty((n_shapes, max_points * 2), dtype=np.dtype('int'))
    shapes[:,0::2] = randint(0, width, size=(n_shapes, max_points))
    shapes[:,1::2] = randint(0, height, size=(n_shapes, max_points))
    points = np.full(n_shapes, min_points)

    colors = randint(0, 256, size=(n_shapes, 4), dtype=np.dtype('uint8'))
    
    return shapes, points, colors

def draw_image(width, height, shapes, points, colors, antialiasing=False):
    '''Draws image from a set of polygons with or without antialiasing'''
    scale = 4 if antialiasing else 1
    image = plImage.new('RGB', (width * scale, height * scale), (255, 255, 255, 0))
    drawer = plDraw.Draw(image, 'RGBA')

    for shape, point, color in zip(shapes, points, colors):
        drawer.polygon((shape[:point * 2] * scale).tolist(), tuple(color))
    if antialiasing: image.thumbnail((width, height))
    return image

def error_abs(a, b):
    '''Calculates difference between two image matrices'''
    return np.abs(np.subtract(a, b, dtype=np.dtype('i4'))).sum()

def error_percent(error, image):
    '''Calculates human-readable % of error from absolute error'''
    return error / (image.shape[0] * image.shape[1] * 255 * 3) * 100

def generate(source, n_shapes, min_points, max_points, internal_res):
    '''Build image. Interrupt program to return current image'''
    population = []

    def changes(shapes, points, colors):
        '''Selects a polygon and randomly executes a change over it'''

        # Configuration for changes
        point_rng = max(width / 2, height / 2)
        shape_rng = max(width / 2, height / 2)
        color_rng = 100
        alpha_rng = 100


        def point(shapes, points, colors, index):
            '''Random change to one point in xy axis'''
            #print("-------------------------------------------------",(index))
            
            shapes = shapes.copy()
            change, point = randint(-point_rng, point_rng+1, size=2), np.random.choice(max_points) * 2

            shapes[index, point:point + 2] = np.clip(shapes[index, point:point + 2] + change, 0, [width, height])
            return shapes, points, colors

        def shape(shapes, points, colors, index):
            '''Random change to a polygon in xy axis'''
            shapes = shapes.copy()
            change = np.tile(randint(-shape_rng, shape_rng+1, size=2), max_points)
            
            shapes[index] = np.clip(shapes[index] + change, 0, boundaries)
            return shapes, points, colors

        def order(shapes, points, colors, index):
            '''Random change to drawing order of a polygon's points'''
            shapes = shapes.copy()
            shuffle = np.random.permutation(points[index])
            shapes[index][0:points[index] *2:2] = shapes[index][0:points[index] *2:2][shuffle]
            shapes[index][1:points[index] *2:2] = shapes[index][1:points[index] *2:2][shuffle]
            return shapes, points, colors

        def number(shapes, points, colors, index):
            '''Change the number of sides of a polygon'''
            points = points.copy()
            if points[index] == min_points:
                points[index] = points[index] + 1
            elif points[index] == max_points:
                points[index] = points[index] - 1
            else:
                points[index] = points[index] + np.random.choice([1, -1])
            return shapes, points, colors

        def color(shapes, points, colors, index):
            '''Random change to the color of a polygon in the RGB axis'''
            colors = colors.copy()
            change = randint(-color_rng, color_rng+1, size=3)

            colors[index][:3] = np.clip(colors[index][:3] + change, 0, 256)
            return shapes, points, colors

        def alpha(shapes, points, colors, index):
            '''Random change to the transparency (alpha layer) of a polygon'''
            colors = colors.copy()
            change = randint(-alpha_rng, alpha_rng+1)

            colors[index][3] = np.clip(colors[index][3] + change, 0, 256)
            return shapes, points, colors

        index, func = randint(n_shapes), np.random.choice([point, shape, order, number, color, alpha])
        return func(shapes, points, colors, index)

    def iterate(shapes, points, colors, image, score):
        '''Makes one change to current set of polygons and returns the best one of the two'''
        new_shapes, new_points, new_colors = changes(shapes, points, colors)
        new_image = np.array(draw_image(width, height, new_shapes, new_points, new_colors))
        new_score = error_abs(internal, new_image)
        #population.append(new_image)
        if new_score <= score:
            return new_shapes, new_points, new_colors, new_image, new_score
        return shapes, points, colors, image, score

    original = plImage.open(source).convert('RGB')
    reduce, restore = resizer(original, internal_res)
    internal, width, height = reduce(original)

    print('Generating {}x{} image with {}x{} internal resolution'.format(*original.size, width, height))

    boundaries = np.tile([width, height], max_points)
    # statistics
    nPopulation = 10
    mutationRate = 0.3
    for i in range (nPopulation):
        shapes, points, colors = initialize(n_shapes, min_points, max_points, width, height)
        boundaries = np.tile([width, height], max_points)
        image = np.array(draw_image(width, height, shapes, points, colors))
        score = error_abs(internal, image)
        population.append([shapes, points, colors, image, score])

    parentsIndex = fitnessProportionalSelection(population, 2)
    #parentsIndex = rankbasedSelection(population, 2)
    #parentsIndex = binaryTournament(population, 2)
    #parentsIndex = randomSelection(population, 2)
    #parentsIndex = truncation(population, 2)

    #crossover
    #print((population))
    children = crossOver(parentsIndex, population, n_shapes, width, height, internal);
    #mutation
    for i in range (len(children)):
        randNo = random()
        if (randNo<mutationRate):
            children[i][0], children[i][1], children[i][2] = changes(children[i][0], children[i][1], children[i][2])
            children[i][3] = np.array(draw_image(width, height, children[i][0], children[i][1], children[i][2]))
            children[i][4] = error_abs(internal, children[i][3])
    #calculating scores
    children[0][3] = np.array(draw_image(width, height, children[0][0], children[0][1], children[0][2]))
    children[0][4] = error_abs(internal, children[0][3])
    children[1][3] = np.array(draw_image(width, height, children[1][0], children[1][1], children[1][2]))
    children[1][4] = error_abs(internal, children[1][3])
    #adding children to population with all stats
    population.append(children[0])
    population.append(children[1])

    

    print('{:>12}  {:>12}  {:>12}  {:>12}'.format('iteration', 'error %', 'error abs', 'avg vert'))

    try:
        it = 0
        while True:
            if not it % 10000: # prints info each 10k iterations
                print('{:>12}  {:>11.4f}%  {:>12}  {:>12.2f}'.format(
                    it, error_percent(score, internal), score, points.mean()))
            shapes, points, colors, image, score = iterate(shapes, points, colors, image, score)
            it += 1
    except:
        print('Generation finished at {} iterations.'.format(it))
        restore(shapes, points, colors).save('evolisa.png', 'PNG')

if __name__ == '__main__':
    from sys import argv

    error_args = (
        "Invalid arguments. Usage:\n"
        "python evolisa.py [source] [# shapes] [min sides] [max sides] [internal res]\n"
        "Example: python evolisa.py source.jpg 50 3 20 160"
        )

    try:
        source= "monkey.jpg"
        n_shapes =50# int(argv[2])
        min_points, max_points =3,20# int(argv[3]), int(argv[4])
        internal_res =160# int(argv[5])
    except:
        print(error_args)

    generate(source, n_shapes, min_points, max_points, internal_res)
        
