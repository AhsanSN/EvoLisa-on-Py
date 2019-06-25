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
    fitness = []
    for i in range(len(population)):
        fitness.append(population[i][4])
    for i in range (1,nSelect+1):
        nthLargestElement = nth_largest(i, fitness)
        for j in range (len(fitness)):
            if (nthLargestElement == population[j][4]):
                selectedIndex.append(j)
                break
    return selectedIndex

# crossover [shapes, points, colors, image, score]
def crossOver(parentsIndex, population, n_shapes, width, height, internal):
    startCopyIndex = randint(10, 25) 
    finishCopyIndex = randint(35, 40)  
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
    i = 0
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
    
    startCopyIndex = randint(10, 25) 
    finishCopyIndex = randint(35, 40)
    
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

    
    #[shapes, points, colors, image, score]
    '''
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
    errorS = np.abs(np.subtract(a, b, dtype=np.dtype('i4'))).sum()
    return errorS

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
            shapes[index][point:point + 2] = np.clip(shapes[index][point:point + 2] + change, 0, [width, height])
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
        if new_score <= score:
            return new_shapes, new_points, new_colors, new_image, new_score
        return shapes, points, colors, image, score

    original = plImage.open(source).convert('RGB')
    reduce, restore = resizer(original, internal_res)
    internal, width, height = reduce(original)

    print('Generating {}x{} image with {}x{} internal resolution'.format(*original.size, width, height))

    # statistics
    nPopulation = 20
    mutationRate = .3
    nChildren = 4 #must be even
    #generating random population
    boundaries = np.tile([width, height], max_points)
    for i in range (nPopulation):
        shapes, points, colors = initialize(n_shapes, min_points, max_points, width, height)
        image = np.array(draw_image(width, height, shapes, points, colors))
        score = error_abs(internal, image)
        population.append([shapes, points, colors, image, (10000000000/score)])

    #printing scores
    for i in range (nPopulation):
        print(population[i][4])

    #showing initial population
    '''
    for i in range (nPopulation):
        restore(np.array(population[i][0]), np.array(population[i][1]), np.array(population[i][2])).save('.\img\initPop_g'+str("")+'_s'+str((population[i][4]))+'.png', 'PNG')
    '''
    print('{:>12}  {:>12}  {:>12}  {:>12}'.format('iteration', 'error %', 'error abs', 'avg vert'))
    bestPop = []
    try:
        it = 0
        while (True):#for k in range (89090900): #
            '''
            if((it>1000)and(it<1010)):
                restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')

            if((it>3000)and(it<3010)):
                restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')

            if((it>5000)and(it<5010)):
                restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')

            if((it>10000)and(it<10010)):
                restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')
            '''
            bestIndex = truncation(population, 1)
            bestPop.append([it, population[bestIndex[0]][4]])
                
            if not it % 500: # prints info each -- iterations
                bestIndex = truncation(population, 1)
                print("iteration:",it,"best score;",(max(population, key=lambda x: x[4]))[4])#population[bestIndex[0]][4])
            if not it % 2000: # prints info each -- iterations
                print("new best image saved")
                restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')
            it += 1

            #find best

            #crossover
            
            for childGeneration in range (nChildren//2):
                #select parents
                parentsIndex = fitnessProportionalSelection(population, 2)
                #parentsIndex = rankbasedSelection(population, 2)
                #parentsIndex = binaryTournament(population, 2)
                #parentsIndex = randomSelection(population, 2)
                #parentsIndex = truncation(population, 2)

            
                children = crossOver(parentsIndex, population, n_shapes, width, height, internal);
            
                #calculating scores
                children[0][3] = np.array(draw_image(width, height, children[0][0], children[0][1], children[0][2]))
                children[0][4] = (10000000000/(error_abs(internal, (children[0][3]))))
                
                children[1][3] = np.array(draw_image(width, height, children[1][0], children[1][1], children[1][2]))
                children[1][4] = (10000000000/(error_abs(internal, (children[1][3]))))
                
                #mutation
                '''
                for i in range (len(children)):
                    randNo = random()
                    if (randNo<mutationRate):
                        children[i][0], children[i][1], children[i][2] = changes(children[i][0], children[i][1], children[i][2])
                        children[i][3] = np.array(draw_image(width, height, children[i][0], children[i][1], children[i][2]))
                        children[i][4] = (1000)000000/(error_abs(internal, children[i][3])))
                    else:
                        children[i][3] = np.array(draw_image(width, height, children[i][0], children[i][1], children[i][2]))
                        children[i][4] = (10000000000/(error_abs(internal, children[i][3])))
                
                #calculating scores
                children[0][3] = np.array(draw_image(width, height, children[0][0], children[0][1], children[0][2]))
                children[0][4] = (10000000000/(error_abs(internal, (children[0][3]))))
                
                children[1][3] = np.array(draw_image(width, height, children[1][0], children[1][1], children[1][2]))
                children[1][4] = (10000000000/(error_abs(internal, (children[1][3]))))
                '''
                
                #print("child scores", children[0][4], children[1][4])
                #adding children to population with all stats
                population.append(children[0])
                population.append(children[1])
                
                if(np.array_equal(children[0][4],children[1][4]) and np.array_equal(children[0][3],children[1][3])):
                    '''
                    print("child1", children[0][3])
                    print("")
                    print("child2", children[1][3])
                    '''
                    1;
                    #print("disaster-------------------------------------")
                
            
            #select new population
            #populationIndices = fitnessProportionalSelection(population, nPopulation)
            #populationIndices = rankbasedSelection(population, nPopulation)
            populationIndices = binaryTournament(population, nPopulation)
            #populationIndices = randomSelection(population, nPopulation)
            #populationIndices = truncation(population, nPopulation)

            #restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\evolisa_'+str(it)+'.png', 'PNG')

            tempPopulation = []
            for i in (populationIndices):
                tempPopulation.append(population[i])
            population = tempPopulation
            '''
            print("stats")
            for i in range (nPopulation):
                print(population[i][4])
            '''
            '''
            print("population scores +++++")
            for i in range (nPopulation):
                print(population[i][4])
            print("population scores -----")
            '''
    except Exception  as  e:
        print(e)
        print('Generation finished at {} iterations.'.format(it))
        #[shapes, points, colors, image, score]
        #print(len(population[0][0]), len(population[0][1]), len(population[0][2]))
        #print("1111111111",shapes)
        #print("1111111111",population[0][0])
        #restore(shapes, points, colors).save('evolisa.png', 'PNG')
        restore(np.array(population[bestIndex[0]][0]), np.array(population[bestIndex[0]][1]), np.array(population[bestIndex[0]][2])).save('.\img\pic_g'+str(it)+'_s'+str(population[bestIndex[0]][4])+'.png', 'PNG')
        print("stats")
        for i in range (nPopulation):
            print(population[i][4])
        
if __name__ == '__main__':
    from sys import argv

    error_args = (
        "Invalid arguments. Usage:\n"
        "python evolisa.py [source] [# shapes] [min sides] [max sides] [internal res]\n"
        "Example: python evolisa.py source.jpg 50 3 20 160"
        )

    try:
        source= "person_png.png"
        n_shapes =50# int(argv[2])
        min_points, max_points =3,20# int(argv[3]), int(argv[4])
        internal_res =200# int(argv[5])
    except:
        print(error_args)

    generate(source, n_shapes, min_points, max_points, internal_res)
        

'''
Basic code taken from: https://github.com/RobertoIA/EvoLisa
'''
