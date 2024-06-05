# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve
import warnings


class FootballMatchPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = StandardScaler()
        self.default_favourite = 'D'

    def preprocess_data(self):
        
        # Create target variable
        
        preds = self.data[['B365H', 'B365A', 'B365D']].idxmin(axis=1)
        # Set predicted result based on max probability index
        self.data.loc[preds == 'B365D', 'FTR_pred'] = 'D'
        self.data.loc[preds == 'B365A', 'FTR_pred'] = 'A'
        self.data.loc[preds == 'B365H', 'FTR_pred'] = 'H'
          
        self.data.loc[self.data['FTR_pred'] != self.data['FTR'], 'Target'] = 1
        self.data['Target'] = self.data['Target'].fillna(0)
        
        # Key: Choose subset where home or away team have played 5 matches, enough data collected at this point
        self.data['num_matches_home'] = self.data['home_wins'] + self.data['home_draws'] + self.data['home_losses']
        self.data['num_matches_away'] = self.data['away_wins'] + self.data['away_draws'] + self.data['away_losses']

        self.data = self.data[(self.data['num_matches_home'] >= 5) | (self.data['num_matches_away'] >= 5)]

        ### Balance dataset ###
        
        # Target: self.data['Target']
        minority_val = self.data['Target'].value_counts(ascending=1).index[0]
        majority_val = self.data['Target'].value_counts(ascending=0).index[0]

        minority_num = self.data['Target'].value_counts(ascending=1).values[0]
        minority_subset = self.data.loc[self.data["Target"] == minority_val, :]
        right_subset = self.data.loc[self.data["Target"] == majority_val, :].sample(minority_num)

        self.data = pd.concat([minority_subset, right_subset], ignore_index=True)

        self.data['B365_home_prob'] = 1 / self.data['B365H']
        self.data['B365_draw_prob'] = 1 / self.data['B365D']
        self.data['B365_away_prob'] =  1 / self.data['B365A']

        # Engineer features
        self.data['rank_diff'] = self.data['home_rank'] - self.data['away_rank']
        self.data['av_goals_scored_home'] = self.data['home_goals_for'] / self.data['num_matches_home']
        self.data['av_goals_conc_home'] = self.data['home_goals_against'] / self.data['num_matches_home']
        self.data['av_goals_scored_away'] = self.data['away_goals_for'] / self.data['num_matches_away']
        self.data['av_goals_conc_away'] = self.data['away_goals_against'] / self.data['num_matches_away']

        # Prepare data for modeling
        self.X = self.data[['rank_diff', 'home_ppg_home', 'away_ppg_away', 'home_wins', 'away_wins', 'home_draws', 'away_draws',
                        'av_goals_scored_home', 'av_goals_conc_home', 'av_goals_scored_away', 'av_goals_conc_away', 'home_losses',
                        'away_losses', 'home_goals_for', 'away_goals_for', 'home_goals_against', 'away_goals_against',
                           'last_5_home_D', 'last_5_home_L', 'last_5_home_W', 'last_5_away_D', 'last_5_away_L', 'last_5_away_W']]

        self.y = self.data['Target']

        # # Scale features
        self.scaler.fit(self.X)
        X_scaled = self.scaler.transform(self.X)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.2)

    def train_model(self, n_splits=5, final=False):
        if final:
            rfc = RandomForestClassifier()
            param_grid = {'n_estimators': [50, 100, 200],
                          'max_depth': [3, 5, None]}

            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(rfc, param_grid, cv=n_splits)
            grid_search.fit(self.scaler.transform(self.X), self.y)

            # Train random forest model with best hyperparameters
            self.model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                                max_depth=grid_search.best_params_['max_depth'])
            self.model.fit(self.scaler.transform(self.X), self.y)
            return

        # Define random forest model and parameter grid for hyperparameter tuning
        rfc = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100, 200],
                      'max_depth': [3, 5, None]}

        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(rfc, param_grid, cv=n_splits)
        grid_search.fit(self.X_train, self.y_train)

        # Train random forest model with best hyperparameters
        self.model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                            max_depth=grid_search.best_params_['max_depth'])
        self.model.fit(self.X_train, self.y_train)

        # Evaluate model performance using cross-validation
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=n_splits)
        # print("Cross-validation scores:", scores)
        print("Average cross-validation score:", np.mean(scores))

    def evaluate_model(self):
        # Test model on testing set
        accuracy = self.model.score(self.X_test, self.y_test)
        print("Test Accuracy:", accuracy, "\n")
        
        # f1_score
        predictions = self.model.predict(self.X_test)
        print(f"f1 score: {f1_score(self.y_test, predictions, average='weighted')}")
        
        
        return accuracy

    def predict_matches(self, matches):
        """ Returns the predicted outcome for a premier league match"""

        # Engineer features
        matches['rank_diff'] = matches['home_rank'] - matches['away_rank']
        matches['num_matches_home'] = matches['home_wins'] + matches['home_draws'] + matches['home_losses']
        matches['num_matches_away'] = matches['away_wins'] + matches['away_draws'] + matches['away_losses']

        matches['av_goals_scored_home'] = matches['home_goals_for'] / matches['num_matches_home']
        matches['av_goals_conc_home'] = matches['home_goals_against'] / matches['num_matches_home']
        matches['av_goals_scored_away'] = matches['away_goals_for'] / matches['num_matches_away']
        matches['av_goals_conc_away'] = matches['away_goals_against'] / matches['num_matches_away']

        # For early on in season, `av_goals_scored` columns will be nan, so impute with zero for now
        ### Only these columns are nan, so can go general fillna(0) on whole dataframe
        matches = matches.fillna(0)
        
        matches['B365_home_prob'] = 1 / matches['B365H']
        matches['B365_draw_prob'] = 1 / matches['B365D']
        matches['B365_away_prob'] =  1 / matches['B365A']

        # Scale features
        X_scaled = self.scaler.transform(matches[['rank_diff', 'home_ppg_home', 'away_ppg_away', 'home_wins', 'away_wins', 'home_draws', 'away_draws',
                        'av_goals_scored_home', 'av_goals_conc_home', 'av_goals_scored_away', 'av_goals_conc_away', 'home_losses',
                        'away_losses', 'home_goals_for', 'away_goals_for', 'home_goals_against', 'away_goals_against',
                        'last_5_home_D', 'last_5_home_L', 'last_5_home_W', 'last_5_away_D', 'last_5_away_L', 'last_5_away_W']])

        # Predict match outcomes. Output bet365 prediction & our prediction
        b365_probs = matches[['B365_home_prob', 'B365_draw_prob', 'B365_away_prob']]
        max_prob_indices = b365_probs.idxmax(axis=1)

        # Set predicted result based on max probability index
        matches.loc[max_prob_indices == 'B365_home_prob', 'b365_predicted_result'] = 'H'
        matches.loc[max_prob_indices == 'B365_draw_prob', 'b365_predicted_result'] = 'D'
        matches.loc[max_prob_indices == 'B365_away_prob', 'b365_predicted_result'] = 'A'


        matches['Prediction'] = self.model.predict(X_scaled)
        matches['Prediction'] = matches['Prediction'].replace({1: 'Lay b365', 0: 'Back b365'})
        matches['RightConf'] = [x[0] for x in self.model.predict_proba(X_scaled)]
        matches['WrongConf'] = [x[1] for x in self.model.predict_proba(X_scaled)]        
        
        prediction_scores = self.model.predict_proba(X_scaled)
        
        return matches[['HomeTeam', 'AwayTeam', 'Prediction', 'b365_predicted_result', 'WrongConf', 'RightConf']]   
    
    
    def plot_confusion_matrix(self):
        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.model.classes_)
        disp.plot()
        plt.show()