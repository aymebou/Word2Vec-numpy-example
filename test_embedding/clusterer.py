from random import randint, choices
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

class Clusterer:

    @staticmethod
    def computeCentroid(points):
        if len(points) == 0:
            return None
        dimension = len(points[0])
        centroid = []
        for d in range(dimension):
            centroid.append(sum([i[d] for i in points]) / len(points))
        return centroid

    @staticmethod
    def norm(p1, p2):
        return cosine(p1, p2)

    def closestCentroidIndex(self, point, centroids):
        distance = self.norm(point, centroids[0])
        closestCentroidIndex = 0
        for i in range(len(centroids)):
            currentNorm = self.norm(point, centroids[i])
            if distance > currentNorm:
                distance = currentNorm
                closestCentroidIndex = i
        return closestCentroidIndex

    def computeCategories(self, dataset, centroids):
        categories = [[] for i in range(len(centroids))]
        for point in dataset:
            closestIndex = self.closestCentroidIndex(point, centroids)
            categories[closestIndex].append(point)
        return categories

    def labelData(self, dataset, centroids):
        return [self.closestCentroidIndex(point, centroids) for point in dataset]

    def computeCentroids(self, categories):
        centroids = []
        for points in categories:
            centroids.append(self.computeCentroid(points))
        return centroids

    def convergenceReached(self, oldCentroids, newCentroids, threshold=0.01):
        for i in range(len(oldCentroids)):
            if self.norm(oldCentroids[i], newCentroids[i]) > threshold:
                return False
        return True

    def instanciateKMeansPPCentroids(self, dataset, k):
        centroids = []
        centroids.append(dataset[randint(0, len(dataset)) - 1])
        probabilityDistribution = self.normalize([self.norm(i, centroids[0]) for i in dataset])
        for centerIndex in range(0, k - 1):
            centroids.append(choices(dataset, probabilityDistribution)[0])
            distances = self.normalize([self.norm(i, centroids[-1]) for i in dataset])
            probabilityDistribution = [min([distances[i], probabilityDistribution[i]]) for i in range(len(dataset))]
        return centroids

    def predictLabels(self, dataset, k, algorithm, initialValues=[]):
        if algorithm == 'kmeans':
            categories, centroids, iterations = self.kMeansPlusPlus(dataset, k)
        elif algorithm == 'kmedoids':
            categories, centroids, iterations = self.kMeansFull(dataset, k, initialValues)
        else:
            return
        return np.array(self.labelData(dataset, centroids)), np.array(centroids)

    def selectRandomUniquePointsInDataset(self, dataset, k):
        chosen = []
        centroids = []
        for i in range(k):
            chosenCentroidIndex = randint(0, len(dataset) - 1)
            while (chosenCentroidIndex in chosen):
                chosenCentroidIndex = randint(0, len(dataset) - 1)
            chosen.append(chosenCentroidIndex)
            centroids.append(dataset[chosenCentroidIndex])
        return centroids

    def kMeansFull(self, dataset, k, initialValues=[]):
        iterations = 0
        if initialValues == []:
            oldCentroids = self.selectRandomUniquePointsInDataset(dataset, k)
        else:
            oldCentroids = initialValues
        print('Starting Iterating on centroids')
        categories = self.computeCategories(dataset, oldCentroids)
        newCentroids = self.computeCentroids(categories)
        while not (self.convergenceReached(oldCentroids, newCentroids)):
            iterations += 1
            print('K-means iteration :', iterations+1)
            oldCentroids = newCentroids
            categories = self.computeCategories(dataset, newCentroids)
            newCentroids = self.computeCentroids(categories)
        print('Finished iterating')
        return categories, newCentroids, iterations

    def normalize(self, array):
        total = sum(array)
        return [i / total for i in array]

    def kMeansPlusPlus(self, dataset, k):
        initialCentroids = self.instanciateKMeansPPCentroids(dataset, k)
        print('Instanciated K-means++ centroids')
        return self.kMeansFull(dataset, k, initialCentroids)

    ##MEDOIDS
    def computePartialCost(self, categories, medoids):
        cost = 0
        for point in medoids:
            cost += self.norm(categories, medoids)
        return cost

    def computeCost(self, categories, medoids):
        totalCost = 0
        for i in range(len(categories)):
            totalCost += self.computePartialCost(categories[i], medoids[i])
        return totalCost

    @staticmethod
    def swap(categories, medoids, categorieIndex, medoidIndex):
        medoid = medoids[medoidIndex]
        medoids[medoidIndex] = categories[categorieIndex]
        categories[categorieIndex] = medoid
        return categories, medoids

    def kMedoids(self, dataset, k, initialValues=[]):
        if initialValues == []:
            medoids = self.selectRandomUniquePointsInDataset(dataset, k)
        else:
            medoids = initialValues
        categories = self.computeCategories(dataset, medoids)
        cost = self.computeCost(categories, medoids)
        costDecreased = True
        iteration = 0
        while costDecreased:
            iteration += 1
            costDecreased = False
            for categorieIndex in range(len(categories)):
                for medoidIndex in range(len(medoids)):
                    categories, medoids = self.swap(categories, medoids, categorieIndex, medoidIndex)
                    swapCost = self.computeCost(categories, medoids)
                    if swapCost >= cost:
                        categories, medoids = self.swap(categories, medoids, categorieIndex, medoidIndex)
                    else:
                        cost = swapCost
                        costDecreased = True
        return medoids, categories, iteration

    def kMedoidsPlusPlus(self, dataset, k):
        initialValues = self.instanciateKMeansPPCentroids(dataset, k)
        return self.kMedoids(dataset, k, initialValues)
