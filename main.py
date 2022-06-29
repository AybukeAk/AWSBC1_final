from datetime import timedelta
import datetime as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

from joblib import dump, load
import pandas as pd
clf = load('filename.joblib')
print(clf)
df = pd.read_csv(r'./data/hotel_bookings.csv')


# encoding
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Feature Extracting
def family(data):
    if ((data['adults'] > 0) & (data['children'] > 0)):
        val = 1
    elif ((data['adults'] > 0) & (data['babies'] > 0)):
        val = 1
    else:
        val = 0
    return val


def feature_eng(df):
    filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
    df = df[~filter]
    df.drop(columns=['company', 'reservation_status', 'agent'], axis=1, inplace=True)
    df.dropna(subset=['country', 'children'], axis=0, inplace=True)
    df.drop(df[df['adr'] < 0].index, inplace=True, axis=0)
    df[df['distribution_channel'] == 'Undefined']
    df.drop(df[df['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

    df['new_is_family'] = df.apply(family, axis=1)

    df.drop(df[df.assigned_room_type == 'L'].index, inplace=True)
    df.drop(df[df.reserved_room_type == 'L'].index, inplace=True)

    unique_room_list = list(
        df.groupby(by='assigned_room_type').agg({'adr': 'mean'}).sort_values(by='adr', ascending=False).index)
    mapper = {}
    k = 10
    for index, i in enumerate(unique_room_list):
        mapper[unique_room_list[index]] = k
        k = k - 1

    df['assigned_room_type'].replace(mapper, inplace=True)
    df['reserved_room_type'].replace(mapper, inplace=True)

    df['new_room_difference'] = df['reserved_room_type'] - df['assigned_room_type']
    df['new_total_people'] = df['adults'] + df['children'] + df['babies']
    df['new_total_stay_day'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    from calendar import month_name
    sorted_months = list(month_name)[1:]

    mapper = {}
    for index, i in enumerate(sorted_months):
        mapper[i] = index + 1

    df['new_month'] = df.arrival_date_month.replace(mapper)
    df['new_month'].value_counts()
    df[['new_month', 'arrival_date_month']]

    df.columns
    cols = ['arrival_date_day_of_month', 'new_month', 'arrival_date_year']
    df['new_arrival_date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis=1)
    df['new_arrival_date'].dtype
    df['new_arrival_date'] = pd.to_datetime(df['new_arrival_date'])

    list_PMS_date = []
    for i in range(df.shape[0]):
        list_PMS_date.append(df.new_arrival_date.iloc[i] - timedelta(days=int(df.lead_time.iloc[i])))

    df["new_PMS_entering_date"] = list_PMS_date
    df['new_special_req_status'] = np.where(df['total_of_special_requests'] != 0, "Yes", "No")
    df['new_dist_channel_type'] = np.where(df['distribution_channel'] != "Direct", "Others", "Direct")

    df['new_room_difference_cat'] = np.nan
    df.loc[df[df['new_room_difference'] > 0].index, 'new_room_difference_cat'] = 1
    df.loc[df[df['new_room_difference'] < 0].index, 'new_room_difference_cat'] = -1
    df.loc[df[df['new_room_difference'] == 0].index, 'new_room_difference_cat'] = 0

    df.drop(df[(df['customer_type'] == 'Group') & (df['new_total_people'] == 1)].index, axis=0, inplace=True)

    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['reservation_status_date'] = df['reservation_status_date'].map(dt.datetime.toordinal)
    df['new_arrival_date'] = df['new_arrival_date'].map(dt.datetime.toordinal)
    df['new_PMS_entering_date'] = df['new_PMS_entering_date'].map(dt.datetime.toordinal)

    df['new_is_weekend'] = np.where([(df['stays_in_weekend_nights'] > 0) & (df['stays_in_week_nights'] == 0)], 1, 0)[0]
    df['new_is_weekday'] = np.where([(df['stays_in_weekend_nights'] == 0) & (df['stays_in_week_nights'] > 0)], 1, 0)[0]
    df['new_is_weekend_and_weekdays'] = \
    np.where([(df['stays_in_weekend_nights'] > 0) & (df['stays_in_week_nights'] > 0)], 1, 0)[0]
    df['new_want_parking_space'] = np.where(df['required_car_parking_spaces'] > 0, 1, 0)
    df['new_special_req_status'] = np.where(df['new_special_req_status'] == 'Yes', 1, 0)
    df['new_adr_per_person'] = df['adr'] / (df['adults'] + df['children'])

    df.drop(columns=['adults', 'babies', 'children', 'total_of_special_requests'], inplace=True, axis=1)

    df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
                     'stays_in_weekend_nights', 'stays_in_week_nights', 'required_car_parking_spaces',
                     'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'reservation_status_date',
                     'is_canceled'], axis=1, inplace=True)
    # encoding
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

    for col in binary_cols:
        label_encoder(df, col)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    ohe_cols = [col for col in cat_cols if 12 >= df[col].nunique() > 2]
    ohe_cols.remove('new_room_difference_cat')

    df = one_hot_encoder(df, ohe_cols, drop_first=True)
    le = LabelEncoder()

    # There are more than 300 classes, so I wanted to use label encoder on this feature.
    df['country'] = le.fit_transform(df['country'])

    return df

user_info = {"is_canceled": 0,
    "hotel": "Resort Hotel",
            "lead_time" : 342,
            "arrival_date_year" : 2015,
            "arrival_date_month" : "July",
            "arrival_date_week_number" : 27,
            "arrival_date_day_of_month" : 1,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 0,
            "adults": 2,
            "children": 0,
            "babies" : 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "C",
            "booking_changes": 3,
            "deposit_type": "No Deposit",
            "agent": " ",
            "company": " ",
            "days_in_waiting_list": 0,
            "customer_type": "Transient",
            "adr": 0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "reservation_status": "Check-Out",
            "reservation_status_date": "2015-07-01"}


print(type(user_info.get('required_car_parking_spaces')))

user = pd.DataFrame(user_info,index = [119390])
new_df = pd.concat([df,user])
new_df = feature_eng (new_df)
user1 = pd.DataFrame(new_df.loc[[119390]])
def result(user1):
    flag = clf.predict(user1)
    if flag==0:
        return ("OKAY")
    else:
        return ("CANCELED")