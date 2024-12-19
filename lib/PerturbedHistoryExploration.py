import numpy as np

class PHEStruct:
    def __init__(self, num_arm, alpha):
        self.dim = num_arm
        self.alpha = alpha

        self.UserArmMean = np.zeros(self.dim)
        self.UserArmTrials = np.zeros(self.dim)

    def updateParameters(self, articlePicked_id, reward):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + reward) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        articlePicked = None
        maxPTA = float('-inf')

        for article in pool_articles:
            if self.UserArmTrials[article.id] == 0:
                return article
            
            perturb = np.sum(np.random.binomial(1, 0.5, int(self.alpha * self.UserArmTrials[article.id])))
            article_pta = (self.UserArmMean[article.id] * self.UserArmTrials[article.id] + float(perturb)) / ((1 + self.alpha) * self.UserArmTrials[article.id])
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class PerturbedHistoryExploration:
    def __init__(self, num_arm, alpha):
        self.users = {}
        self.num_arm = num_arm
        self.alpha = alpha
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm, self.alpha)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, reward, userID):
        self.users[userID].updateParameters(articlePicked.id, reward)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean