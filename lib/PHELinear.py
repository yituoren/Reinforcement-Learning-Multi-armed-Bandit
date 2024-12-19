import torch
import numpy as np

class PHELStruct:
    def __init__(self, featureDimension, n_articles, lambda_, alpha):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.dim = featureDimension
        self.A = lambda_ * torch.eye(self.dim, device=self.device)
        self.AInv = torch.eye(self.dim, device=self.device) / lambda_
        self.lambda_ = lambda_
        self.alpha = alpha
        self.history = []
        self.UserTheta = torch.zeros(self.dim, device=self.device)
        self.UserArmTrials = torch.zeros(n_articles, device=self.device)
        self.num = n_articles
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, reward):
        self.time += 1
        feature_vector = torch.tensor(articlePicked_FeatureVector, device=self.device).float()
        self.A += torch.outer(feature_vector, feature_vector)
        self.history.append((feature_vector, reward))
        b = torch.zeros(self.dim, device=self.device)
        for x, r in self.history:
            b += (r + torch.bernoulli(0.5 * torch.ones(self.alpha, device=self.device)).sum()) * x
        # self.AInv = torch.inverse(self.A * (1 + self.alpha))
        AInvFeature = torch.matmul(self.AInv, feature_vector)
        self.AInv -= torch.outer(AInvFeature, AInvFeature) / (1 + torch.dot(feature_vector, AInvFeature))
        self.UserTheta = torch.matmul(self.AInv, b) / (1 + self.alpha)

    def getTheta(self):
        b = torch.zeros(self.dim, device=self.device)
        for x, r in self.history:
            b += r * x
        theta = torch.matmul(torch.inverse(self.A), b)
        return np.array(theta.cpu())

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        articlePicked = None
        maxPTA = float('-inf')

        if self.time < self.dim and self.time < self.num and self.time < len(pool_articles):
            for article in pool_articles:
                if self.UserArmTrials[article.id] == 0:
                    articlePicked = article
                    break
        else:
            for article in pool_articles:
                feature_vector = torch.tensor(article.featureVector, device=self.device).float()
                article_pta = torch.dot(self.UserTheta, feature_vector)
                if maxPTA < article_pta:
                    articlePicked = article
                    maxPTA = article_pta

        self.UserArmTrials[articlePicked.id] += 1
        return articlePicked

class PHELinear:
    def __init__(self, dimension, n_articles, lambda_, alpha):
        self.users = {}
        self.dim = dimension
        self.lambda_ = lambda_
        self.alpha = alpha
        self.num = n_articles
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHELStruct(self.dim, self.num, self.lambda_, self.alpha)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, reward, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dim], reward)

    def getTheta(self, userID):
        return self.users[userID].getTheta()