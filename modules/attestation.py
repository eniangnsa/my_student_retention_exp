import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class Attestation:
    def __init__(self, attest_path, target_path):
        self.attest_path = Path(attest_path)
        self.target_path = Path(target_path)
        #  get a list of files in the directory
        self.attest_list = list(self.attest_path.glob("*.xlsx"))

        # read all the files into a single dataframe
        self.df_list = [pd.read_excel(file) for file in self.attest_list]
        self.attest_data = pd.concat(self.df_list, ignore_index=True)
        self.target_data = pd.read_csv(target_path)
        self.passed = ['зачтено', 'академическая разница', 'отлично', 'хорошо', 'удовлетворительно']
        self.not_passed = ['Неявка', 'Не зачтено', 'неудовлетворительно']
        self.attest_data.drop(['Unnamed: 1', 'Unnamed: 16',
                               'Unnamed: 2', 'Unnamed: 14', 'Unnamed: 15',
                               'Unnamed: 4', 'Unnamed: 12', 'Unnamed: 13',
                               'Unnamed: 0', 'Unnamed: 10', 'Unnamed: 11',
                               'Unnamed: 3', 'Unnamed: 8', 'Unnamed: 9',
                               'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1, inplace=True)
        # rename fields
        self.new_col_names = {
            "НСИ ИД": 'student_id',
            "GUIDЗачетной книги": "record_book",
            "GUIDУчебного плана": "study_plan",
            "Период сдачи": "period",
            "Дисциплина": "discipline",
            "Вид контроля": "test_type",
            "Период контроля": "test_period",
            "Порядковый номер периода контроля": "Semester",
            "Учебный год": "study_year",
            "Полугодие": "half_year",
            "Отметка": "grade",
            "Тип ведомости": "type_grade_report",
            "Есть выборы": "has_choice",
            "Выбрана": "chosen"
        }
        self.attest_data.rename(columns=self.new_col_names, inplace=True)
        self.attest_data['period'] = pd.to_datetime(self.attest_data['period'], dayfirst=True, errors='coerce')

    def preprocess(self):
        self.target_data['start_date'] = pd.to_datetime(self.target_data['start_date'])
        self.target_data['end_date'] = pd.to_datetime(self.target_data['end_date'])
        self.target_data['global_start_date'] = pd.to_datetime(self.target_data['global_start_date'])
        self.target_data['id'] = self.target_data.index

    def points_from_grade(self, value):
        if value == "отлично":
            return 5
        elif value == "хорошо":
            return 4
        elif value == "удовлетворительно":
            return 3
        elif value == "неудовлетворительно":
            return 2
        else:
            return 0

    def zachot_points(self, value):
        if value in self.passed:
            return 1
        elif value in self.not_passed:
            return 0
        else:
            return value

    def filter_data(self):
        target_ids = set(self.target_data['student_id'].unique())
        attest_ids = set(self.attest_data['student_id'].unique())
        self.inner_ids = target_ids & attest_ids

        # get the matching ids from the inner_ids
        matching_targets = self.target_data.loc[self.target_data['student_id'].isin(self.inner_ids)]
        # print(matching_targets.columns)
        matching_targets = matching_targets[['id', 'student_id', 'global_start_date', 'end_date']]
        

        # merge the matching ids of the targets with attest_data
        joined_data = pd.merge(matching_targets, self.attest_data, on='student_id', how='left')

        # Filter the records for the desired period of interest
        filtered_data = joined_data[(joined_data['period'] > joined_data['global_start_date']) & (
                joined_data['period'] < joined_data['end_date'])]
        # drop duplicates
        filtered_data.drop_duplicates(inplace=True)

        return filtered_data

    def extract_features(self, target_path: Path):
        self.target_path = target_path
        self.target_data = pd.read_csv(self.target_path)
        self.preprocess()
        filtered_data = self.filter_data()
        if filtered_data.shape[0] == 0:
            return pd.DataFrame()
        # get test type features
        test_type_count = filtered_data.groupby(['id', 'test_type']).size().reset_index(name='count')
        test_type_count = test_type_count.pivot_table(index='id', columns='test_type', values='count', fill_value=0)

        # rename columns for easy reference
        test_type_cols = {
            "Выпускная квалификационная работа": "thesis",
            "Государственный экзамен": "state_exam",
            "Дифференцированный зачет": "diff_zachot",
            "Зачет": "zachot",
            "Контрольная работа": "test",
            "Курсовая работа": "course_work",
            "Курсовой проект": "course_project",
            "Реферат": "abstract",
            "Экзамен": "exam"
        }

        test_type_count.rename(columns=test_type_cols, inplace=True)

        # let's extract some features from grades
        grade_count = filtered_data.groupby(['id', 'grade']).size().reset_index(name='count')
        grade_count = grade_count.pivot_table(index='id', columns='grade', values='count', fill_value=0)

        # again rename the columns
        grade_cols = {
            "Не выбрал": "not_chosen",
            "Не зачтено": "not_passed",
            "Неявка": "absent",
            "академическая разница": "academic_diff",
            "зачтено": "passed",
            "неудовлетворительно": "unsatisfactory",
            "отлично": "excellent",
            "удовлетворительно": "satisfactory",
            "хорошо": "good"
        }
        grade_count.rename(columns=grade_cols, inplace=True)
        for col in grade_count.columns:
            if col not in grade_cols.values():
                del grade_count[col]

        # filter based on exam to calculate the gpa
        exam_filter = filtered_data[filtered_data['test_type'] == "Экзамен"]
        exam_filter['points'] = exam_filter['grade'].apply(self.points_from_grade)

        student_gpa = exam_filter.groupby(['id'])[['points']].mean()
        student_gpa.rename(columns={"points": "GPA"}, inplace=True)

        # filter for zachot and extract features there
        zachot_filter = filtered_data[(filtered_data['test_type'] == "Зачет") & (filtered_data['grade'] != "Не выбрал")]
        # create a column to store the points from zachots
        zachot_filter['zachot_points'] = zachot_filter['grade'].apply(self.zachot_points)
        zachot_filter['zachot_points'] = pd.to_numeric(zachot_filter['zachot_points'], errors='coerce')
        zachot_gpa = zachot_filter.groupby(['id'])[['zachot_points']].mean()
        zachot_gpa.rename(columns={"zachot_points": "zachot_gpa"}, inplace=True)

        # for subjects that are optional or not
        optional = zachot_filter.groupby(['id', 'has_choice']).size().reset_index(name='count')
        optional = optional.pivot_table(index='id', columns='has_choice', fill_value=0, values='count')
        optional.rename(columns={
            "Да": "optional_subject_zachot",
            "Нет": "not_optional_subject"
        }, inplace=True)

        # for subjects that were chosen
        chosen_subject = zachot_filter.groupby(['id', 'chosen']).size().reset_index(name='count')
        chosen_subject = chosen_subject.pivot_table(index='id', columns='chosen', fill_value=0, values='count')
        chosen_subject.rename(columns={
            "Да": "chosen_subject_zachot",
            "Нет": "not_chosen_subject_zachot"
        }, inplace=True)

        # features from the target_data
        features = self.target_data.loc[self.target_data['student_id'].isin(self.inner_ids)].copy()

        dates = ['start_date', 'global_start_date']
        for date in dates:
            features[date + '_month'] = features[date].dt.month
            features[date + '_day'] = features[date].dt.day
            # год не делаем, потому что в дальнейшем придётся переучивать систему
            del features[date]

        features['profile'] = features['profile'].fillna(' ')

        # start merging the data  we have
        features = features.merge(chosen_subject, on='id', how='inner')
        features = features.merge(optional, on='id', how='inner')
        features = features.merge(zachot_gpa, on='id', how='inner')
        features = features.merge(student_gpa, on='id', how='inner')
        features = features.merge(grade_count, on='id', how='inner')
        features = features.merge(test_type_count, on='id', how='inner')

        return features
