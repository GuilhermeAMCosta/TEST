# imports
from os import mkdir
import math
from statistics import median, stdev
from matplotlib import pyplot as plt
from random import uniform, choice, randint, gauss
from time import gmtime, strftime, time
from datetime import datetime
import numpy as np
import cv2
import csv

class DE:

    def __init__(self):
        self.pop = []  # population's positions
        self.m_nmdf = 0.00  # diversity variable
        self.diversity = []
        self.fbest_list = []
        self.ns1 = 0
        self.ns2 = 0
        self.nf1 = 0
        self.nf2 = 0

    def generateGraphs(self, fbest_list, diversity_list, max_iterations, uid, run):
        #graficos
        plt.plot(range(0, max_iterations), fbest_list, 'r--')
        plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'convergence.png')
        plt.clf()
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'diversity.png')
        plt.clf()

    def updateDiversity(self):
        diversity = 0
        aux_1 = 0
        aux2 = 0
        a = 0
        b = 0
        d = 0

        for a in range(0, len(self.pop)):
            b = a + 1
            for i in range(b, len(self.pop)):
                aux_1 = 0

                ind_a = self.pop[a]
                ind_b = self.pop[b]

                for d in range(0, len(self.pop[0])):
                    aux_1 = aux_1 + (pow(ind_a[d] - ind_b[d], 2).real)
                aux_1 = (math.sqrt(aux_1).real)
                aux_1 = (aux_1 / len(self.pop[0]))

                if b == i or aux_2 > aux_1:
                    aux_2 = aux_1
            diversity = (diversity) + (math.log((1.0) + aux_2).real)

        if self.m_nmdf < diversity:
            self.m_nmdf = diversity

        return (diversity / self.m_nmdf).real

    # fitness_function

    def fitness(self, individual):

        #CARREGA IMAGENS
        w, h = template.shape[::-1]

        #INFORMARÇOES DE INDIVIDUO
        x = int(individual[0])
        y = int(individual[1])
        t = individual[2]
        s = individual[3]

        #print("X:", individual[0], "Y:", individual[1],
              #"TETA:", individual[2], "S:", individual[3])

        if(x > width_temp/2 and x < width-width_temp/2 and y > height_temp and y < height - height_temp/2
            and s > 0.5 and s < 1.5 and t > -60 and t < 60):
            #CORTA IMAGEM
            crop_img = img[y:y + h, x:x + w]
            R = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), t, s)
            output = cv2.warpAffine(crop_img, R, (img.shape[1], img.shape[0]))
            #ORB
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(output, None)

            if des2 is None:
                return 0
            elif(des1 is None):
                return 0
            else:
                #BFMatcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
                matches = bf.match(des1,des2)
                #result = len(matches)
                cont = 0
                for i in range(0, len(matches)):
                    #print(matches[i].distance)
                    dist = matches[i].distance
                    #print(dist)
                    if dist < TH:
                       cont = cont+1
                result = cont
        else:
            result = 0

        #print(result)
        return (result)

    def generatePopulation(self, pop_size, dim, bounds):
        for ind in range(pop_size):
            lp = []
            for d in range(dim):
                lp.append(uniform(bounds[d][0], bounds[d][1]))
            self.pop.append(lp)

    def evaluatePopulation(self):
        fpop = []
        for ind in self.pop:
            fpop.append(self.fitness(ind))
        #print("FPOP:", fpop)
        return fpop

    def getBestSolution(self, maximize, fpop):
        fbest = fpop[0]
        best = [values for values in self.pop[0]]
        #print("Best:", best, fpop[0])
        for ind in range(1, len(self.pop)):
            if maximize == True:
                if fpop[ind] >= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]
            else:
                if fpop[ind] <= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]

        return fbest, best

    def rand_1_bin(self, ind, dim, wf, cr):
        p1 = ind
        while (p1 == ind):
            p1 = choice(self.pop)
        p2 = ind
        while (p2 == ind or p2 == p1):
            p2 = choice(self.pop)
        p3 = ind
        while (p3 == ind or p3 == p1 or p3 == p2):
            p3 = choice(self.pop)

        # print('current: %s\n' % str(ind))
        # print('p1: %s\n' % str(p1))
        # print('p2: %s\n' % str(p2))
        # print('p3: %s\n' % str(p3))
        # input('...')

        cutpoint = randint(0, dim - 1)
        candidateSol = []

        # print('cutpoint: %i' % (cutpoint))
        # input('...')

        for i in range(dim):
            if (i == cutpoint or uniform(0, 1) < cr):
                candidateSol.append(p3[i] + wf * (p1[i] - p2[i]))  # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
            else:
                candidateSol.append(ind[i])

        # print('candidateSol: %s' % str(candidateSol))
        # input('...')
        # print('\n\n')
        return candidateSol

    def currentToBest_2_bin(self, ind, best, dim, wf, cr):
        p1 = ind
        while (p1 == ind):
            p1 = choice(self.pop)
        p2 = ind
        while (p2 == ind or p2 == p1):
            p2 = choice(self.pop)

        # print('current: %s\n' % str(ind))
        # print('p1: %s\n' % str(p1))
        # print('p2: %s\n' % str(p2))
        # input('...')

        cutpoint = randint(0, dim - 1)
        candidateSol = []

        # print('cutpoint: %i' % (cutpoint))
        # input('...')

        for i in range(dim):
            if (i == cutpoint or uniform(0, 1) < cr):
                candidateSol.append(ind[i] + wf * (best[i] - ind[i]) + wf * (
                            p1[i] - p2[i]))  # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        # print('candidateSol: %s' % str(candidateSol))
        # input('...')
        # print('\n\n')
        return candidateSol

    def boundsRes(self, ind, bounds):
        for d in range(len(ind)):
            if ind[d] < bounds[d][0]:
                ind[d] = bounds[d][0]
            if ind[d] > bounds[d][1]:
                ind[d] = bounds[d][1]

    def diferentialEvolution(self, pop_size, dim, bounds, max_iterations, runs, maximize=True, p1=0.5, p2=0.5,
                             learningPeriod=50, crPeriod=5, crmUpdatePeriod=25):
        # generete execution identifier
        mkdir(str(uid))
        mkdir(str(uid) + '/graphs')
        # to record the results
        results = open(str(uid) + '/results.txt', 'a')
        records = open(str(uid) + '/records.txt', 'a')
        results.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(uid), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        results.write(
            '=================================================================================================================\n')
        records.write('ID: %s\tDate: %s\tRuns: %s\n' % (str(uid), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs)))
        records.write(
            '=================================================================================================================\n')
        avr_fbest_r = []
        avr_diversity_r = []
        fbest_r = []
        best_r = []
        elapTime_r = []

        # runs
        for r in range(runs):
            elapTime = []
            start = time()
            records.write('Run: %i\n' % r)
            records.write('Iter\tGbest\tAvrFit\tDiver\tETime\t\n')

            # start the algorithm
            best = []  # global best positions
            fbest = 0.00

            # global best fitness
            if maximize == True:
                fbest = 0.00
            else:
                fbest = math.inf

            # initial_generations
            self.generatePopulation(pop_size, dim, bounds)
            fpop = self.evaluatePopulation()

            fbest, best = self.getBestSolution(maximize, fpop)
            #print("Best Solution:", fbest, best)

            # evolution_step
            # generates crossover rate values
            crm = 0.5
            crossover_rate = [gauss(crm, 0.1) for i in range(pop_size)]
            cr_list = []
            for iteration in range(max_iterations):
                avrFit = 0.00
                # #update_solutions
                strategy = 0
                for ind in range(0, len(self.pop)):
                    # generate weight factor values
                    weight_factor = gauss(0.5, 0.3)
                    if uniform(0, 1) < p1:
                        candSol = self.rand_1_bin(self.pop[ind], dim, weight_factor, crossover_rate[ind])
                        strategy = 1
                    else:
                        candSol = self.currentToBest_2_bin(self.pop[ind], best, dim, weight_factor, crossover_rate[ind])
                        strategy = 2

                    self.boundsRes(candSol, bounds)
                    fcandSol = self.fitness(candSol)

                    if maximize == False:
                        if fcandSol <= fpop[ind]:
                            self.pop[ind] = candSol
                            fpop[ind] = fcandSol
                            cr_list.append(crossover_rate[ind])
                            if strategy == 1:
                                self.ns1 += 1
                            elif strategy == 2:
                                self.ns2 += 1
                        else:
                            if strategy == 1:
                                self.nf1 += 1
                            elif strategy == 2:
                                self.nf2 += 1
                    else:
                        if fcandSol >= fpop[ind]:
                            self.pop[ind] = candSol
                            fpop[ind] = fcandSol
                            cr_list.append(crossover_rate[ind])
                            if strategy == 1:
                                self.ns1 += 1
                            elif strategy == 2:
                                self.ns2 += 1
                        else:
                            if strategy == 1:
                                self.nf1 += 1
                            elif strategy == 2:
                                self.nf2 += 1

                    avrFit += fpop[ind]
                avrFit = avrFit / pop_size
                self.diversity.append(self.updateDiversity())

                fbest, best = self.getBestSolution(maximize, fpop)

                self.fbest_list.append(fbest)
                elapTime.append((time() - start) * 1000.0)
                records.write('%i\t%.4f\t%.4f\t%.4f\t%.4f\n' % (iteration, round(fbest, 4), round(avrFit, 4), round(self.diversity[iteration], 4), elapTime[iteration]))

                if iteration % crPeriod == 0 and iteration != 0:
                    crossover_rate = [gauss(crm, 0.1) for i in range(pop_size)]
                    if iteration % crmUpdatePeriod == 0:
                        crm = sum(cr_list) / len(cr_list)
                        cr_list = []

                if iteration % learningPeriod == 0 and iteration != 0:
                    p1 = (self.ns1 * (self.ns2 + self.nf2)) / (
                                self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2))
                    p2 = 1 - p1
                    self.nf2 = 0
                    self.ns1 = 0
                    self.ns2 = 0
                    self.nf1 = 0

            records.write('Pos: %s\n\n' % str(best))
            fbest_r.append(fbest)
            best_r.append(best)
            elapTime_r.append(elapTime[max_iterations - 1])
            self.generateGraphs(self.fbest_list, self.diversity, max_iterations, uid, r)
            avr_fbest_r.append(self.fbest_list)
            avr_diversity_r.append(self.diversity)

            self.pop = []
            self.m_nmdf = 0.00
            self.diversity = []
            self.fbest_list = []
            p1 = p2 = 0.5
            self.nf2 = 0
            self.ns1 = 0
            self.ns2 = 0
            self.nf1 = 0

        fbestAux = [sum(x) / len(x) for x in zip(*avr_fbest_r)]
        diversityAux = [sum(x) / len(x) for x in zip(*avr_diversity_r)]
        self.generateGraphs(fbestAux, diversityAux, max_iterations, uid, 'Overall')
        records.write(
            '=================================================================================================================')
        if maximize == False:
            results.write('Gbest Overall: %.4f\n' % (min(fbest_r)))
            results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(min(fbest_r))]))
        else:
            results.write('Gbest Overall: %.4f\n' % (max(fbest_r)))
            results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(max(fbest_r))]))

        results.write('Gbest Average: %.4f\n' % (sum(fbest_r) / len(fbest_r)))
        results.write('Gbest Median: %.4f)\n' % (median(fbest_r)))
        if runs > 1:
            results.write('Gbest Standard Deviation: %.4f\n\n' % (stdev(fbest_r)))
        results.write('Elappsed Time Average: %.4f\n' % (sum(elapTime_r) / len(elapTime_r)))
        if runs > 1:
            results.write('Elappsed Time Standard Deviation: %.4f\n' % (stdev(elapTime_r)))
        results.write(
            '=================================================================================================================\n')

        #print(best_r)
        print("\nID:", uid)
        print("Melhor Semelhança:", max(fbest_r))
        print("Melhores Pontos:", best_r[fbest_r.index(max(fbest_r))])

        return (best_r[fbest_r.index(max(fbest_r))])

if __name__ == '__main__':


    #CONATGEM DE TEMPO
    start = time()
    with open('Experiment_0.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            # print(row[1], row[2], row[3], row[8], row[13], row[14], row[19], row[20])
            Test_Name = (row[1])
            Type_of_Test = (row[2])
            SCMFI = (row[3])
            FOI = (row[8])
            L_TOP_X = (row[13])
            L_TOP_Y = (row[14])
            R_BOTTOM_X = (row[19])
            R_BOTTOM_Y = (row[20])

            if (SCMFI != '' and SCMFI != 'SCMFI' and FOI != '' and FOI != 'FOI'):
                # print('Landscape:', SCMFI, '\t', 'Template:', FOI, '\t', 'Type of Test:', Type_of_Test, '\t',
                #       'Illumination:', Test_Name)

                real = cv2.imread(SCMFI)
                real_template = cv2.imread(FOI)

                X = (int(R_BOTTOM_X) - int(L_TOP_X)) / 2 + int(L_TOP_X)
                Y = (int(R_BOTTOM_Y) - int(L_TOP_Y)) / 2 + int(L_TOP_Y)

                # print('Gabarito:', '\t',  'X:', X,'\t', 'Y:', Y)

                # cv2.imshow('oi', real)
                # #cv2.imshow('oi', real_template)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                #IMAGENS COM FILTRO CINZA
                img = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
                template = cv2.cvtColor(real_template, cv2.COLOR_BGR2GRAY)

                #INFORMAÇÕES DE DIMENSÃO DA IMAGEM
                height, width = img.shape[:2]
                height_temp, width_temp = template.shape[:2]
                # print("FEATURES OF LANDSCAPE IMAGE:\nHeight = ", height, "\nWidth = ", width, "\n")
                # print("FEATURES OF TEMPLATE IMAGE:\nHeight = ", height_temp, "\nWidth = ", width_temp,
                #     '\n----------------------------------\nTo crop:\n',
                #     (width_temp / 2),
                #     "< x <", (width - (width_temp / 2)), "\n", (height_temp / 2), "< y <", (height - (height_temp / 2)),
                #     "\n----------------------------------\n")

                #TRATAMENTO DA SADE
                max_iterations = 40
                pop_size = 30
                dim = 4
                runs = 10
                bounds = ((width_temp, width - width_temp),(height_temp, height - height_temp), (-60, 60), (0.5, 1.5))
                TH = 29
                uid = datetime.now()
                p = DE()
                verify = 235

                #RETORNO DOS MELHORES PONTOS
                points = p.diferentialEvolution(pop_size, dim, bounds, max_iterations, runs, maximize=True)
                x = int(points[0])
                y = int(points[1])
                print("Pontos Encontrados:", x,'\t', y)
                #MOSTRA RESULTADOS
                a = int(height_temp / 2)
                b = int(width_temp / 2)
                #MANIPULA IMAGEM PARA EXIBIÇÃO
                cv2.rectangle(real, (x, y), (x+width_temp, y+height_temp), (0, 255, 0), 3)
                new_im1 = cv2.resize(real, (600, 500), interpolation=cv2.INTER_AREA)
                new_im2 = cv2.resize(real_template, (600, 500), interpolation=cv2.INTER_AREA)
                numpy_horizontal = np.hstack((new_im1, new_im2))
                cv2.imwrite(str(uid) + '/real.png', numpy_horizontal)
                #VERIFICA RESULTADO
                dif_X = X - x
                dif_Y = Y - y
                if(np.abs(dif_X) < verify and np.abs(dif_Y) < verify):
                    print('Encontrado!', dif_X, dif_Y)
                else:
                    print('Não Encontrado!', dif_X, dif_Y)

end = time()
time = end - start
results = open(str(uid) + '/results.txt', 'a')
results.write("Tempo de Execução: %s segundos" % time)
