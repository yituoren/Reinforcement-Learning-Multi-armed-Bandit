import torch
import numpy as np

class TSLStruct:
    def __init__(self, featureDimension, lambda_, standard_deviation):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.dim = featureDimension
        self.A = lambda_ * torch.eye(self.dim, device=self.device)
        self.AInv = torch.eye(self.dim, device=self.device) / lambda_
        self.lambda_ = lambda_
        self.b = torch.zeros(self.dim, device=self.device)
        self.UserTheta = torch.zeros(self.dim, device=self.device)
        self.sigma = standard_deviation

    def updateParameters(self, articlePicked_FeatureVector, reward):
        feature_vector = torch.tensor(articlePicked_FeatureVector, device=self.device).float() / self.sigma
        self.A += torch.outer(feature_vector, feature_vector)
        self.b += (feature_vector * reward) / self.sigma
        # self.AInv = torch.inverse(self.A)
        AInvFeature = torch.matmul(self.AInv, feature_vector)
        self.AInv -= torch.outer(AInvFeature, AInvFeature) / (1 + torch.dot(feature_vector, AInvFeature))
        self.UserTheta = torch.matmul(self.AInv, self.b)

    def getTheta(self):
        return np.array(self.UserTheta.cpu())

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        articlePicked = None
        maxPTA = float('-inf')

        for article in pool_articles:
            feature_vector = torch.tensor(article.featureVector, device=self.device).float()
            article_pta = torch.normal(torch.dot(self.UserTheta, feature_vector), torch.sqrt(torch.dot(torch.matmul(self.AInv, feature_vector), feature_vector)))
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class TSLinear:
    def __init__(self, dimension, lambda_, standard_deviation):
        self.users = {}
        self.dim = dimension
        self.lambda_ = lambda_
        self.standard_deviation = standard_deviation
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TSLStruct(self.dim, self.lambda_, self.standard_deviation)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, reward, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dim], reward)

    def getTheta(self, userID):
        return np.array(self.users[userID].UserTheta.cpu())