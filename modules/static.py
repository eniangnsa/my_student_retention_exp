import pandas as pd
from pathlib import Path

import warnings

warnings.filterwarnings('ignore')


class Static:
    def __init__(self, static_path, target_path):
        self.static_path = Path(static_path)
        # self.target_path = Path(target_path)
        self.static_data = pd.read_excel(self.static_path, header=2)
        self.target_data = pd.read_csv(target_path)
        # self.target_data = pd.read_csv(self.target_path)
        self.rename_cols()
        self.static_data['office_enrollment_date'] = pd.to_datetime(self.static_data['office_enrollment_date'],
                                                                    dayfirst=True)
        self.static_data['year_enrollment'] = pd.to_datetime(self.static_data['year_enrollment'], dayfirst=True)
        self.static_data['DOB'] = pd.to_datetime(self.static_data['DOB'], dayfirst=True)
        self.static_data['age_at_enrollment'] = (self.static_data['office_enrollment_date'] - self.static_data[
            'DOB']).dt.days // 365
        # self.preprocess()

    # rename the fields of the input data
    def rename_cols(self):
        rename_cols = {
            "ТГУ_НСИ_Ид": "student_id",
            "ДатаРождения": "DOB",
            "ГодПоступления": "year_enrollment",
            "УровеньПодготовки": "edu_level",
            "СпециальностьНаименование": "spec_name",
            "СпециальностьКодСпециальности": "spec_code",
            "Профиль": "profile",
            "Поступил": "enrolled",
            "ОснованиеПоступления": "funding",
            "ФормаОбучения": "edu_form",
            "Предмет1": "subject_1",
            "Предмет2": "subject_2",
            "Предмет3": "subject_3",
            "Оценка1": "grade_1",
            "Оценка2": "grade_2",
            "Оценка3": "grade_3",
            "ИндивидуальныеДостижения": "individual_achievement",
            "БезВступительныхИспытаний": "no_entrance_test",
            "СтатусЛицаБВИ": "status_person_bwi",
            "Олимпиада": "olympiad",
            "ОснованиеПриемаБВИ": "basis_of_acceptance_bwi",
            "Льгота": "benefit",
            "Страна": "country",
            "Регион": "region",
            "Представление": "address",
            "КанцелярскийНомерПриказаОЗачислении": "office_enrollment_order",
            "КанцелярскаяДатаПриказаОЗачислении": "office_enrollment_date"
        }
        return self.static_data.rename(columns=rename_cols, inplace=True)

    # preprocess the fields
    def preprocess(self):
        self.target_data['start_date'] = pd.to_datetime(self.target_data['start_date'])
        self.target_data['global_start_date'] = pd.to_datetime(self.target_data['global_start_date'])
        self.target_data['end_date'] = pd.to_datetime(self.target_data['end_date'])
        self.target_data['id'] = self.target_data.index

    # filter by the desired time
    def filter_data(self):
        target_id = set(self.target_data['student_id'].unique())
        static_id = set(self.static_data['student_id'].unique())
        self.inner_id = target_id & static_id

        new_target_data = self.target_data.loc[self.target_data['student_id'].isin(self.inner_id)]
        new_target_data = new_target_data[['id', 'student_id', 'global_start_date', 'end_date']]

        joined_data = pd.merge(new_target_data, self.static_data, on='student_id', how='left')
        # filter for the required period
        filtered_data = joined_data[joined_data['office_enrollment_date'] <= joined_data['end_date']]

        # some preprocessing
        filtered_data['subject_1'] = filtered_data['subject_1'].fillna('').astype(str) + " "
        filtered_data['subject_2'] = filtered_data['subject_2'].fillna('').astype(str) + " "
        filtered_data['subject_3'] = filtered_data['subject_3'].fillna('').astype(str) + " "
        filtered_data['spec_name'] = filtered_data['spec_name'].fillna('').astype(str) + " "

        return filtered_data

    # extract features
    def get_features(self, target_path):
        self.target_data = pd.read_csv(target_path)
        self.preprocess()
        self.filtered_data = self.filter_data()
        #  let's find the mean entrance score
        self.filtered_data['mean_grade'] = self.filtered_data[['grade_1', 'grade_2', 'grade_3']].mean(axis=1)

        # create a new field for subjects
        self.filtered_data['subjects'] = self.filtered_data['subject_1'] + self.filtered_data['subject_2'] + \
                                         self.filtered_data['subject_3']
        # new feature from subjects
        self.subjects = self.filtered_data.groupby(['id'])['subjects'].sum()

        # get features for spec_name
        self.spec_names = self.filtered_data.groupby(['id'])['spec_name'].sum()

        # number of unique enrollment year
        self.num_unique_enrollment_year = self.filtered_data.groupby(['id'])[['year_enrollment']].nunique()
        self.num_unique_enrollment_year.rename(columns={"year_enrollment": "num_unique_enrollment_year"}, inplace=True)

        # number of enrollment for each  student
        self.num_enrolled = self.filtered_data.groupby(['id', 'enrolled']).size().reset_index(name='count')
        self.num_enrolled = self.num_enrolled.pivot_table(index='id', columns='enrolled', fill_value=0, values='count')
        self.num_enrolled.rename(columns={
            "Да": "num_times_enrolled",
            "Нет": "num_times_not_enrolled"
        }, inplace=True)

        # number of unique education level
        self.num_unique_edu_level = self.filtered_data.groupby(['id'])[['edu_level']].nunique()
        self.num_unique_edu_level.rename(columns={
            'edu_level': "num_unique_edu_level"}, inplace=True)

        # average time spent in days for each student from enrollment date
        self.sorted_data = self.filtered_data.sort_values(['id', 'office_enrollment_date'])
        self.sorted_data['time_spent_days'] = self.sorted_data.groupby(['id'])[
            'office_enrollment_date'].diff().dt.days.fillna(0)
        self.avg_time_spent_per_student = self.sorted_data.groupby(['id'])[['time_spent_days']].mean()
        self.avg_time_spent_per_student.rename(columns={'time_spent_days': "avg_time_spent_days"}, inplace=True)

        # total time spent in days for each student from enrollment date
        self.total_time_spent_per_student = self.sorted_data.groupby(['id'])[['time_spent_days']].sum()
        self.total_time_spent_per_student.rename(columns={"time_spent_days": "total_time_spent_days"}, inplace=True)

        # demographic data about the student
        sorted_data = self.filtered_data.sort_values(['id', 'year_enrollment'], ascending=False)
        self.demographic_data = sorted_data[
            ["id", "age_at_enrollment", "year_enrollment", "country", "enrolled", "mean_grade"]].drop_duplicates(
            subset="id")

        # features from target_data
        features = self.target_data.loc[self.target_data['student_id'].isin(self.inner_id)].copy()

        dates = ['start_date', 'global_start_date']
        for date in dates:
            features[date + '_month'] = features[date].dt.month
            features[date + '_day'] = features[date].dt.day
            del features[date]

        features['profile'] = features['profile'].fillna(" ")

        # merge all features into one dataframe
        features = features.merge(self.num_unique_enrollment_year, on='id', how='inner')
        features = features.merge(self.num_enrolled, on='id', how='inner')
        # features = features.merge(self.num_unique_edu_level, on='id', how='inner')
        features = features.merge(self.total_time_spent_per_student, on='id', how='inner')
        features = features.merge(self.avg_time_spent_per_student, on='id', how='inner')
        features = features.merge(self.demographic_data, on='id', how='inner')
        features = features.merge(self.subjects, on='id', how='inner')
        features = features.merge(self.spec_names, on='id', how='inner')

        # drop some uninformative field
        # features = features.drop(["end_date", "student_id", "id"], axis=1)
        features = features.drop(["year_enrollment"], axis=1)

        # handle missing values in country
        features['country'] = features['country'].fillna(" ")

        return features
