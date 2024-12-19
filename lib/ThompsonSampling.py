import numpy as np

class TSStruct:
    def __init__(self, num_arm, standard_deviation):
        self.dim = num_arm
        self.sigma = standard_deviation
        self.var = standard_deviation * standard_deviation

        self.UserArmMean = np.zeros(self.dim)
        self.UserArmTrials = np.zeros(self.dim)
        self.UserArmVar = np.ones(self.dim)
    
    def updateParameters(self, articlePicked_id, reward):
        self.UserArmMean[articlePicked_id] = (self.var * self.UserArmMean[articlePicked_id] + self.UserArmVar[articlePicked_id] * reward) / (self.var + self.UserArmVar[articlePicked_id])
        self.UserArmVar[articlePicked_id] = self.var * self.UserArmVar[articlePicked_id] / (self.var + self.UserArmVar[articlePicked_id])

    def getTheta(self):
        return self.UserArmMean
    
    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            # article_pta = np.random.normal(self.UserArmMean[article.id], self.sigma)
            article_pta = np.random.normal(self.UserArmMean[article.id], np.sqrt(self.UserArmVar[article.id]))
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta
        
        return articlePicked
    
class ThompsonSampling:
    def __init__(self, num_arm, standard_deviation):
        self.users = {}
        self.num_arm = num_arm
        self.CanEstimateUserPreference = False
        self.standard_deviation = standard_deviation
    
    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TSStruct(self.num_arm, self.standard_deviation)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, reward, userID):
        self.users[userID].updateParameters(articlePicked.id, reward)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean