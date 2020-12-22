
import time
import datetime
from contextlib import contextmanager
import math
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import f1_score
# from sklearn.preprocessing import normalize


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    # >>> with measure_time('Heavy computation'):
    # >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end - start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter)


def same_team_(sender, player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)


def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"]
    X_pairs = pd.DataFrame(data=np.zeros((n_ * 22, len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_ * 22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        # other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j)]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
                # print(int(player_j == y_.iloc[i]))
            idx += 1

    return X_pairs, y_pairs


def compute_distance_(X_):
    d = np.sqrt((X_["x_sender"] - X_["x_j"]) ** 2 + (X_["y_sender"] - X_["y_j"]) ** 2)
    return d


def compute_distance_2_(x_A, x_B, y_A, y_B):
    d = np.sqrt((x_A - x_B) ** 2 + (y_A - y_B) ** 2)
    return d


def dist_(X_, index_, i, j):
    x_i = X_.iat[index_, (i * 2)]
    x_j = X_.iat[index_, (j * 2)]
    y_i = X_.iat[index_, (i * 2) + 1]
    y_j = X_.iat[index_, (j * 2) + 1]

    d = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    return d


def score_(X_, index_, receiver):
    sender = X_.iloc[index_].sender
    players = np.arange(1, 23)
    sc = 0
    for player_j in players:
        sc += (same_team_(receiver, player_j) * (-2) + 1) * dist_(X_, index_, receiver, player_j)

    return sc


def dist_df(X_):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_8", "d_9", "d_10", "d_10", "d_12",
                        "d_13", "d_14", "d_15", "d_16", "d_17", "d_18", "d_19", "d_20", "d_21", "d_22"]
    X_dist = pd.DataFrame(data=np.zeros((n_, len(pair_feature_col))), columns=pair_feature_col)
    X_LS_pairs, _ = make_pair_of_players(X_)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)

    for i in range(n_):
        X_dist.at[i, "sender"] = X_.at[i, "sender"]
        idx = 0
        for j in pair_feature_col:
            if j != "sender":
                X_dist.at[i, j] = X_LS_pairs.at[(22 * i) + idx, "distance"]
                idx += 1
    return X_dist


def sc_df(X_):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_8", "d_9", "d_10", "d_10", "d_12",
                        "d_13", "d_14", "d_15", "d_16", "d_17", "d_18", "d_19", "d_20", "d_21", "d_22"]
    X_sc_df = pd.DataFrame(data=np.zeros((n_, len(pair_feature_col))), columns=pair_feature_col)
    X_LS_pairs, _ = make_pair_of_players(X_)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)

    for i in range(n_):
        X_sc_df.at[i, "sender"] = X_.at[i, "sender"]
        idx = 0
        for j in pair_feature_col:
            if j != "sender":
                X_sc_df.at[i, j] = score_(X_, i, idx)
                idx += 1
    return X_sc_df


def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = 3000
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples, 22))
        for i in range(n_samples):
            probas[i, predictions[i] - 1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples,))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1, 23)[mask]
            predictions[i] = int(selected_players[0])

    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1, 23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1] + "\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1, 23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1] + "\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1] + "\n")

    return file_name


def divide_df(df, k, i):
    div_len = int(len(df.index) / k)
    if i == 0:
        df_i = df.iloc[:div_len, :]
        df_reste = df.iloc[div_len + 1:, :]
    elif i == k - 1:
        df_i = df.iloc[div_len * (k - 1) + 1:, :]
        df_reste = df.iloc[:div_len * (k - 1), :]
    else:
        df_i = df.iloc[i * div_len + 1:(i + 1) * div_len, :]
        new_df_1 = df.iloc[:i * div_len, :]
        new_df_2 = df.iloc[(i + 1) * div_len + 1:, :]
        df_reste = new_df_1.append(new_df_2, ignore_index=True, sort=False)

    return df_i, df_reste


def divide(data, k, i):
    """
    Divides the dataset into k parts,
    output = two arrays, one is the i^th part the other one is the given dataset w/o i^th part

    :param k:   int < n
                number of divisions
    :param data: array of length n
    :param i:   int < k
                i^th part
    :return:

    l_sample:     array of length n/k
                i^th part of the dataset
    t_sample:     array of length n-(n/k)
                remaining data as one dataset
    """

    new_data = np.array_split(data, k)
    l_sample = new_data[i]
    t_sample = []
    t_sample_hold = np.delete(new_data, i, axis=0)
    for elements in t_sample_hold:
        t_sample.extend(elements.tolist())

    np.asarray(t_sample)
    return l_sample, t_sample


def kcv_score(mc, X_data, y_data, k=6):
    """
    Gives the mean k-fold cross validation score of dataset
    for a given model classifier

    :param X_data:  array of length n
                    input data

    :param y_data:  array of length n
                    output data

    :param mc:      class
                    model classifier

    :param k:       int < n
                    k-fold cross validation

    :return:

    mean_score:          float
                    mean score value

    """
    score = 0.
    for i in range(0, k, 1):
        X_test, X_train = divide_df(X_data, k, i)
        y_test, y_train = divide_df(y_data, k, i)
        mc2 = mc.fit(X_train, np.ravel(y_train))
        y_pred_test = mc2.predict(X_test)
        y_pred_test = format_proba(y_pred_test)
        score += f1_score(y_test, y_pred_test)
    mean_score = score / k
    return mean_score


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.floor(n * multiplier + 0.5) / multiplier)


def two_nearest_opponents_(X_):
    n = X_.shape[0]
    first_opponent_dis = 0
    second_opponent_dis = 0
    two_opponent_col = ["dis_first_op", "dis_second_op", "dis_first_al", "dis_second_al"]
    X_additional_col = pd.DataFrame(data=np.zeros((n, len(two_opponent_col))), columns=two_opponent_col)

    # Filling the additional column with the two nearest opponents
    receiver_x = 0
    receiver_y = 0

    m = int(n / 22)
    for i in range(m):
        for j in range(22):
            """if X_.iloc[i*22+j]["same_team"] == 0:
                #first opponent is the opponent itself
                first_opponent_dis = 0
                second_opponent_dis = 0
            else :"""
            # Save the player position
            receiver_x = X_.iloc[i * 22 + j]["x_j"]
            receiver_y = X_.iloc[i * 22 + j]["y_j"]

            first_opponent_dis = 10000
            second_opponent_dis = 10000
            first_ally_dis = 10000
            second_ally_dis = 10000
            for k in range(22):
                if X_.iloc[i * 22 + k]["same_team"] == 0:
                    dis = np.sqrt(
                        (receiver_x - X_.iloc[i * 22 + k]["x_j"]) ** 2 + (receiver_y - X_.iloc[i * 22 + k]["y_j"]) ** 2)
                    if second_opponent_dis >= dis > 0.01:
                        second_opponent_dis = dis
                        if first_opponent_dis >= dis > 0.01:
                            second_opponent_dis = first_opponent_dis
                            first_opponent_dis = dis

                if X_.iloc[i * 22 + k]["same_team"] == 1:
                    dis = np.sqrt(
                        (receiver_x - X_.iloc[i * 22 + k]["x_j"]) ** 2 + (
                                  receiver_y - X_.iloc[i * 22 + k]["y_j"]) ** 2)
                    if second_ally_dis >= dis > 0.01:
                        second_ally_dis = dis
                        if first_ally_dis >= dis > 0.01:
                            second_ally_dis = first_ally_dis
                            first_ally_dis = dis

                X_additional_col.iloc[i * 22 + j]["dis_first_al"] = first_ally_dis
                X_additional_col.iloc[i * 22 + j]["dis_second_al"] = second_ally_dis
                X_additional_col.iloc[i * 22 + j]["dis_first_op"] = first_opponent_dis
                X_additional_col.iloc[i * 22 + j]["dis_second_op"] = second_opponent_dis

    # Merge the two dataframe
    result = pd.concat([X_, X_additional_col], axis=1)

    return result


def format_proba(y_):
    n_ = y_.shape[0]
    m_ = int(n_ / 22)
    y_proba = np.zeros(n_)
    for i in range(m_-1):

        hold_list_ = y_[i * 22:(i+1)*22]
        idx = np.argmax(hold_list_)
        hold_list_ = np.zeros(22)
        hold_list_.itemset(idx, 1)

        y_proba[i * 22:(i+1)*22] = hold_list_

    return np.ravel(y_proba)


def clean_vals_(X_):
    n_ = X_.shape[0]
    new_X_ = X_

    for i in range(n_):
        if new_X_.at[i, "distance"] < 0.05:
            new_X_.at[i, "distance"] = 1000000.

    return new_X_


def normalize(X_):
    n_ = X_.shape[0]
    m_ = X_.shape[1]
    X_new = X_
    for i in range(n_):
        row_i = X_[i]
        for j in range(m_):
            if row_i[j] < 0.:
                row_i[j] = 0.
        tot = np.sum(row_i)
        X_new[i] /= tot
    return X_new


if __name__ == '__main__':
    prefix = 'C:/Users/nguye/Desktop/Master_1_Bis/Machine_learning/Project_3/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_features = load_from_csv(prefix + 'X_features2.csv')
    # X_LS = load_from_csv(prefix + 'input_training_set.csv')
    y_LS_pairs = pd.read_csv(prefix + 'y2.csv', usecols=["pass"])
    # X_features = clean_vals_(X_features)


    # Load test data
    X_TS = load_from_csv(prefix + 'input_test_set.csv')
    X_TS_features = load_from_csv("X_TS2.csv")
    # X_TS_features = clean_vals_(X_TS_features)

    print("Loading data complete")

    # Build the model

    model = ensemble.RandomForestRegressor(random_state=42)

    with measure_time('Training'):
        print('Training...')
        model.fit(X_features, np.ravel(y_LS_pairs))

    # ------------------------------ Prediction ------------------------------ #


    # Predict
    with measure_time('Predicting'):
        print('Predicting...')
        y_pred = model.predict(X_TS_features)

    print("Prediction complete")

    # Deriving probas

    probas = y_pred.reshape(X_TS.shape[0], 22)
    probas = normalize(probas)
    # Estimated score of the model

    with measure_time('Predicting score'):
        print('Predicting score...')
        predicted_score = kcv_score(model, X_features, y_LS_pairs, k=6)

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="toy_example_probas")
    print('Submission file "{}" successfully written'.format(fname))
