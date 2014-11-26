import numpy
import math
import re
import time
import datetime


class Data(object):
    def __init__(self, user_id, item_id, rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating


def get_data(file_name):
    fin = open(file_name, 'r')
    next(fin)
    res = [Data(*map(int, re.split(",|\n", line)[:3])) for line in fin]
    fin.close()
    return res


data_train = get_data("train.csv")
data_validation = get_data("validation.csv")
dataset = data_train + data_validation  # lets train with validation data set


def get_average_item_rating(dataset):
    sum, count = {}, {}
    for data in dataset:
        sum[data.item_id] = sum.get(data.item_id, 0) + data.rating
        count[data.item_id] = count.get(data.item_id, 0) + data.rating
    return {key: round(float(sum[key]) / float(count[key])) for key in sum}


gamma = 0.005
lam = 0.02
feature_count = 5
feature_default = 0.1


def get_deafult_features():
    return [feature_default] * feature_count


r = {}
average_item_rating = get_average_item_rating(dataset)
bu, bi, q, p, mu = {}, {}, {}, {}, 0


def calc_prediction(user, item):
    if (user not in bu) and (item not in bi):
        score = mu
    elif user not in bu:
        score = mu + bi[item]
    elif item not in bi:
        score = mu + bu[user]
    else:
        score = mu + bi[item] + bu[user] + numpy.dot(q[item], p[user])
    return max(1, min(5, score))


def calc_features(feature1, feature2, error):
    for i, old in enumerate(feature1):
        feature1[i] += gamma * (error * feature2[i] - lam * old)
    return feature1


for data in dataset:
    if data.user_id not in r:
        r[data.user_id] = dict()
    user = r[data.user_id]
    user[data.item_id] = data.rating

for user in r:
    bu[user] = float(0)
    p[user] = get_deafult_features()
    user_ratings = r[user]
    for item in user_ratings:
        bi[item] = float(0)
        if item not in q:
            q[item] = get_deafult_features()
        if not user_ratings.get(item):
            user_ratings[item] = average_item_rating[item]

for _ in range(2):
    print(_)
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    for user in r:
        user_ratings = r[user]
        for item in user_ratings:
            rating = user_ratings[item]
            error = rating - calc_prediction(user, item)
            mu += gamma * (error - lam * mu)
            bu[user] += gamma * (error - lam * bu[user])
            bi[item] += gamma * (error - lam * bi[item])
            old_q = list(q[item])
            old_p = list(p[user])
            q[item] = calc_features(q[item], old_p, error)
            p[user] = calc_features(p[user], old_q, error)

# Checking rmse on validation data set
sse = 0
for data in data_validation:
    sse += (data.rating - calc_prediction(data.user_id, data.item_id)) ** 2
rmse = math.sqrt(sse / len(data_validation))
print(rmse)

test_file = open("test-ids.csv", "r")
test_result = open("result.csv", "w")
next(test_file)
test_result.write("id,rating\n")
for line in test_file:
    test, user, item = map(int, re.split(",|\n", line)[:3])
    rating = calc_prediction(user, item)
    test_result.write("%d,%f\n" % (test, rating))
test_file.close()
test_result.close()