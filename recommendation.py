import numpy as np
import math
import pdb


###################### Preprocessing ######################

print("Preprocessing")

k = 3

complete_set = np.loadtxt('data/u.data')

# Removes timestamp since they aren't necessary
complete_set = complete_set[:,:3]

# Unique user IDs in set
user_ids = np.unique(complete_set[:,0])


# Removes 3 random samples for each user

temp = []

for user_id in user_ids:
    current_indicies = np.where(complete_set[:,0] == user_id)[0]
    shuffled = np.random.shuffle(current_indicies)
    to_be_removed = current_indicies[0:k]
    for j in range(len(to_be_removed)):
        temp.append(to_be_removed[j])

test_set = complete_set[temp]
train_set = np.delete(complete_set,temp,axis=0)
del temp

# Build dict of users and the movies they have rated
user_dict = {}
for user_id, movie_id, rating in complete_set:
    if user_id in user_dict:
        user_dict[user_id][movie_id] = rating
    else:
        user_dict[user_id] = {}
        user_dict[user_id][movie_id] = rating

for user_id in user_ids:
    user_dict[user_id]['avg_rating'] = np.mean(user_dict[user_id].values())

del user_ids, complete_set

# Build dict of movies and users who have rated them
movie_ids = np.unique(train_set[:, 1])
movies_dict = {}

for movie_id in movie_ids:
    indices = np.where(train_set[:, 1] == movie_id)[0]
    list_of_users = train_set[indices,0]
    movies_dict[movie_id] = {'user_list': list_of_users}

print("Preprocessing done")

###################### Preprocessing done ######################

# Takes two movie ID's and returns the set of users who have rated both movies
def set_u(movie_i, movie_j):
    if movie_j not in movies_dict or movie_i not in movies_dict:
        return []
    users_who_rated_i = movies_dict.get(movie_i)['user_list']
    users_who_rated_j = movies_dict.get(movie_j)['user_list']
    return np.intersect1d(users_who_rated_i, users_who_rated_j)

# Takes a user ID and movie ID and returns the rating of the movie, given by user

def rating(user_id, movie_id):
    if movie_id in user_dict[user_id]:
        return user_dict[user_id][movie_id]
    else:
        return 0

# Takes two movie IDs and returns a similarity score of the two movies 
# based on previous users ratings
def sim(movie_i, movie_j):
    intersection = set_u(movie_i, movie_j)
    if len(intersection) == 0: # The two movies have no users in common
        return 0

    first_sum, second_sum, third_sum = 0, 0, 0

    for user in intersection:
        average_rating = user_dict[user]['avg_rating']
        movie_i_rating = rating(user, movie_i)
        movie_j_rating = rating(user, movie_j)
        first_sum += abs((movie_i_rating - average_rating) * (movie_j_rating - average_rating))
        second_sum += (movie_i_rating - average_rating)**2
        third_sum += (movie_j_rating - average_rating)**2

    second_sum = math.sqrt(second_sum)
    third_sum = math.sqrt(third_sum)
    if (first_sum == 0 or second_sum == 0 or third_sum == 0):
        return 0
    return first_sum / (second_sum * third_sum)


# Takes a user ID and movie ID and returns a prediction of 
# how the user would have rated the movie
def predict(user, movie_i):
    movies_rated_by_user = user_dict[user]
    if len(movies_rated_by_user) == 0: # New user - cold start problem
        return 0

    first_sum, second_sum = 0, 0

    for movie_j in movies_rated_by_user:
        similarity = sim(movie_i, movie_j)
        rating = movies_rated_by_user[movie_j]
        first_sum += (rating * similarity)
        second_sum += abs(similarity)
        
    if first_sum == 0 or second_sum == 0:
        return 0

    predicted_value = first_sum / second_sum
    return predicted_value


# Calculates and returns Root Means Squared Error and precision based on test set given.
def RMSE(test_set):
    number_of_wrong_preds, mySum = 0, 0
    n = len(test_set)
    print('Length of test set: ' + str(n))
    for idx, test in enumerate(test_set):
        predicted_val = predict(int(test[0]),int(test[1]))
        if int(round(predicted_val)) != test[2]:
            number_of_wrong_preds += 1
        if predicted_val == 0:
            n = n-1
        mySum += (predicted_val - test[2])**2
        print('Calculated ' + str(idx) + ' out of: ' + str(n))
    finalValue = math.sqrt((1.0/n) * mySum)
    return finalValue, (float(number_of_wrong_preds) / float(n))

rmse, accuracy = RMSE(test_set)
print('Accuracy: ' + str(accuracy))
print('RMSE: '  + str(rmse))