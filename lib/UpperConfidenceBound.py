import numpy as np

class UCBStruct:
    def __init__(self, num_arm, standard_deviation):
        self.dim = num_arm
        self.sigma = standard_deviation

        self.UserArmMean = np.zeros(self.dim)
        self.UserArmTrials = np.zeros(self.dim)

        self.time = 0

    def updateParameters(self, articlePicked_id, reward):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + reward) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1

        self.time += 1
    
    def getTheta(self):
        return self.UserArmMean
    
    def decide(self, pool_articles):
        articlePicked = None
        maxPTA = float('-inf')

        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article
            article_pta = self.UserArmMean[article.id] + self.sigma * np.sqrt(2 * np.log(self.time) / self.UserArmTrials[article.id])
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class UpperConfidenceBound:
    def __init__(self, num_arm, standard_deviation):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False
        self.standard_deviation = standard_deviation
    
    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm, self.standard_deviation)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, reward, userID):
        self.users[userID].updateParameters(articlePicked.id, reward)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean