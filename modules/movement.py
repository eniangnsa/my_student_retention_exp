import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class StudentAnalysis:
    def __init__(self, movement_path, anonymous_path, target_path):
        self.movements_data = pd.read_csv(movement_path, encoding='windows-1251', sep=';')
        self.target_data = pd.read_csv(target_path)
        # self.target_data = pd.read_csv(target_path)
        self.anonymous_data = pd.read_excel(anonymous_path)
        self.anonymous_data.rename(columns={"ФизическоеЛицо": "GUID"}, inplace=True)
        self.anonymous_data['GUID'] = self.anonymous_data['GUID'].apply(self.make_lower)
        self.movements_data['GUID'] = self.movements_data["GUID"].apply(self.make_lower)
        self.movements = pd.merge(self.movements_data, self.anonymous_data, on="GUID", how="inner")
        self.rename_cols = {
            'НСИ_ИД': 'student_id',
            'Дата': 'date',
            'Время': 'time',
            'Корпус': 'building',
            'Направление': 'direction',
            'Допуск': 'access'
        }

        self.movements.rename(columns=self.rename_cols, inplace=True)
        self.movements['date'] = pd.to_datetime(self.movements['date'])

    def make_lower(self, x):
        return str(x).lower()

    def preprocess_data(self):
        self.target_data['start_date'] = pd.to_datetime(self.target_data['start_date'])
        self.target_data['end_date'] = pd.to_datetime(self.target_data['end_date'])
        self.target_data['global_start_date'] = pd.to_datetime(self.target_data['global_start_date'])
        self.target_data['id'] = self.target_data.index
        # del self.movements['Unnamed: 0']

    def classify_building(self, building):
        if building == "Общежитие":
            return "Hostel"
        elif building == "Главный корпус":
            return "Main Building"
        elif building == "Научная Библиотека":
            return "Library"
        elif building == "Центр Культуры":
            return "Cultural Centre"
        elif building == "Спорт.Корпус":
            return "Sport Complex"
        else:
            return "Academic Building"

    def convert_time_to_hours(self, total_time_each_building):
        total_time_each_building['total_time_academic_building'] = total_time_each_building[
                                                                       'total_time_academic_building'] / 3600
        total_time_each_building['total_time_cultural_center'] = total_time_each_building[
                                                                     'total_time_cultural_center'] / 3600
        total_time_each_building['total_time_library'] = total_time_each_building['total_time_library'] / 3600
        total_time_each_building['total_time_sport'] = total_time_each_building['total_time_sport'] / 3600
        total_time_each_building['total_time_main_building'] = total_time_each_building[
                                                                   'total_time_main_building'] / 3600

        return total_time_each_building

    def filter_data(self):
        target_ids = set(self.target_data['student_id'].unique())
        mov_ids = set(self.movements['student_id'].unique())
        self.inner_ids = target_ids & mov_ids

        new_target_data = self.target_data.loc[self.target_data['student_id'].isin(self.inner_ids)]
        # print(new_target_data.columns)
        new_target_data = new_target_data[['id', 'student_id', 'global_start_date', 'end_date']]

        joined_data = pd.merge(new_target_data, self.movements, on='student_id', how='left')
        filtered_data = joined_data[
            (joined_data['date'] > joined_data['global_start_date']) & (joined_data['date'] < joined_data['end_date'])]
        filtered_data['building_type'] = filtered_data['building'].apply(
            lambda building: self.classify_building(building))
        filtered_data['datetime'] = pd.to_datetime(filtered_data['date'].astype(str) + ' ' + filtered_data['time'],
                                                   format='%Y-%m-%d %H:%M:%S')

        return filtered_data

    # extract features
    def extract_features(self, target_path):
        self.target_data = pd.read_csv(target_path)
        self.preprocess_data()
        # get the filtered data first
        filtered_data = self.filter_data()

        if filtered_data.shape[0] == 0:
            return pd.DataFrame()

        # Extract frequency features for each building
        grouped_data = filtered_data.groupby(['id', 'building_type']).size().reset_index(name='count')
        filtered_data_building = grouped_data[grouped_data['building_type'].isin(
            ['Academic Building', 'Hostel', 'Cultural Centre', 'Library', 'Sport Complex', 'Main Building'])]
        freq_in_each_building = filtered_data_building.pivot_table(index=['id'], columns='building_type',
                                                                   values='count', fill_value=0)
        freq_in_each_building = freq_in_each_building.rename(columns={
            "Academic Building": "freq_academic_building",
            "Hostel": "freq_hostel",
            "Library": "freq_library",
            "Cultural Centre": "freq_cultural_centre",
            "Sport Complex": "freq_sport_complex",
            "Main Building": "freq_main_building"
        })

        # Extract features for time spent in each building
        attendance = filtered_data.sort_values(by=['id', 'datetime'])
        attendance['time_spent'] = attendance.groupby(['id'])['datetime'].shift(-1) - attendance['datetime']
        attendance['time_spent'] = attendance['time_spent'].dt.total_seconds().fillna(0)

        total_time = attendance.groupby(["id", "building_type"])['time_spent'].sum().abs().reset_index(
            name="total_time")
        filtered_data_time = total_time[total_time['building_type'].isin(
            ['Academic Building', 'Hostel', 'Cultural Centre', 'Library', 'Sport Complex', 'Main Building'])]
        total_time_each_building = filtered_data_time.pivot_table(index=['id'], columns='building_type',
                                                                  values='total_time', fill_value=0)

        # Rename columns in total_time_each_building before merging
        total_time_each_building = total_time_each_building.rename(columns={
            "Academic Building": "total_time_academic_building",
            "Hostel": "total_time_hostel",
            "Cultural Centre": "total_time_cultural_center",
            "Library": "total_time_library",
            "Sport Complex": "total_time_sport",
            "Main Building": "total_time_main_building"
        })
        #   # drop irrelevant columns
        # total_time_each_building.drop(columns=['Academic Building', 'Cultural Centre', 'Library', 'Main Building', 'Sport Complex'],inplace=True , axis=1)
        total_time_each_building = self.convert_time_to_hours(total_time_each_building)

        # Extract features for the most visited building
        grouped_data = filtered_data.groupby(['id'])['building_type'].value_counts().reset_index(name='most_visited')

        # Find the most frequent building type for each GUID and season
        most_visited = grouped_data.groupby(['id']).apply(lambda x: x.nlargest(1, 'most_visited')).reset_index(
            drop=True)

        most_visited.rename(columns={"building_type": "most_visited",
                                     "most_visited": "most_visited_freq"},
                            inplace=True)

        # get features from the target_data
        features = self.target_data.loc[self.target_data['student_id'].isin(self.inner_ids)].copy()

        dates = ['start_date', 'global_start_date']
        for date in dates:
            features[date + '_month'] = features[date].dt.month
            features[date + '_day'] = features[date].dt.day
            del features[date]

        features['profile'] = features['profile'].fillna(" ")

        #  start merging all the features into one dataframe
        features = features.merge(most_visited, on='id', how='inner')
        features = features.merge(freq_in_each_building, on='id', how='inner')
        features = features.merge(total_time_each_building, on='id', how='inner')

        return features
