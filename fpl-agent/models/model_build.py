import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
# Import model metrics for testing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
# Import models for testings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

class ModelMaker:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.model = None
        self.scaler = None
        
    def fetch_player_data(self):
        """Fetch enhanced player data including more features"""
        bootstrap_static = requests.get(self.base_url + "bootstrap-static/").json()
        elements_df = pd.DataFrame(bootstrap_static['elements'])
        
        # Include more relevant features for analysis
        selected_columns = [
            'id', 'web_name', 'element_type', 'now_cost', 'ep_next',
            'minutes', 'form', 'points_per_game', 'selected_by_percent',
            'transfers_in', 'transfers_out', 'influence', 'creativity',
            'threat', 'ict_index', 'team'
        ]
        # Rename the minutes column to avoid conflict
        elements_df = elements_df[selected_columns].rename(columns={'minutes': 'total_season_minutes'})
        return elements_df

    def add_features(self, df):
        # Rolling averages
        df['rolling_avg_points_3'] = df['rolling_avg_points'].rolling(3, min_periods=1).mean()
        
        # Lag features
        # TODO fix lagged features, they should be for each player over the gameweeks
        df['points_lag_1'] = df['total_points'].shift(1)
        df['minutes_lag_2'] = df['rolling_avg_minutes'].shift(2)
        
        # Fill NaN values for lag/rolling features
        df.fillna(0, inplace=True)
        
        return df

    def fetch_actual_points(self, gw):
        """Fetch detailed match statistics for each player"""
        gw_data = requests.get(f"{self.base_url}event/{gw}/live/").json()
        detailed_stats = []
        
        for p in gw_data['elements']:
            stats = p['stats']
            detailed_stats.append({
                'id': p['id'],
                'total_points': stats['total_points'],
                'gw_minutes': stats['minutes'],  # Renamed to gw_minutes
                'goals_scored': stats.get('goals_scored', 0),
                'assists': stats.get('assists', 0),
                'clean_sheets': stats.get('clean_sheets', 0),
                'goals_conceded': stats.get('goals_conceded', 0),
                'bonus': stats.get('bonus', 0)
            })
            
        return pd.DataFrame(detailed_stats)

    def collect_data(self, gameweeks):
        """Collect enhanced dataset with historical performance"""
        players_df = self.fetch_player_data()
        all_data = []
        
        for gw in gameweeks:
            actual_stats = self.fetch_actual_points(gw)
            gw_data = players_df.merge(actual_stats, on='id', how='left')
            gw_data['gameweek'] = gw
            
            # Calculate rolling averages
            if len(all_data) > 0:
                previous_data = pd.concat(all_data)
                rolling_stats = previous_data.groupby('id').agg({
                    'total_points': ['mean', 'std'],
                    'gw_minutes': 'mean',  # Using renamed column
                    'bonus': 'mean'
                }).reset_index()
                
                # Flatten the column names
                rolling_stats.columns = ['id', 'rolling_avg_points', 'points_std', 
                                       'rolling_avg_minutes', 'rolling_avg_bonus']
                gw_data = gw_data.merge(rolling_stats, on='id', how='left')
            
            all_data.append(gw_data)
        
        return pd.concat(all_data, ignore_index=True)
    
    def assess_data_quality(self, df):
        """Perform comprehensive data quality checks"""
        total_rows = len(df)
        quality_report = {
            'total_rows': total_rows,
            'nan_analysis': {},
            'zero_analysis': {},
            'negative_analysis': {},
            'column_stats': {}
        }
        
        # Check for NaN values in each column
        nan_counts = df.isna().sum()
        for column in df.columns:
            nan_count = nan_counts[column]
            if nan_count > 0:
                quality_report['nan_analysis'][column] = {
                    'count': int(nan_count),
                    'percentage': round((nan_count / total_rows) * 100, 2)
                }
        
        # Check for zeros and negatives in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            # Zero analysis
            zero_count = (df[column] == 0).sum()
            if zero_count > 0:
                quality_report['zero_analysis'][column] = {
                    'count': int(zero_count),
                    'percentage': round((zero_count / total_rows) * 100, 2)
                }
            
            # Negative value analysis
            if df[column].min() < 0:
                neg_count = (df[column] < 0).sum()
                quality_report['negative_analysis'][column] = {
                    'count': int(neg_count),
                    'percentage': round((neg_count / total_rows) * 100, 2),
                    'min_value': float(df[column].min())
                }
            
            # Basic statistics
            quality_report['column_stats'][column] = {
                'mean': float(df[column].mean()) if not df[column].isna().all() else None,
                'median': float(df[column].median()) if not df[column].isna().all() else None,
                'std': float(df[column].std()) if not df[column].isna().all() else None,
                'min': float(df[column].min()) if not df[column].isna().all() else None,
                'max': float(df[column].max()) if not df[column].isna().all() else None
            }
        
        return quality_report

    def clean_data_for_analysis(self, df):
            """Clean data based on quality assessment"""
            df_clean = df.copy()
            
            # List of essential columns that must not contain NaN
            essential_columns = ['total_points', 'ep_next', 'gw_minutes', 'form', 
                            'points_per_game', 'ict_index']
            
            # Drop rows where essential columns have NaN
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=essential_columns)
            
            # Ensure numeric columns are properly typed
            numeric_columns = [
                'total_points', 'ep_next', 'gw_minutes', 'form', 
                'points_per_game', 'ict_index', 'now_cost', 'goals_scored',
                'assists', 'clean_sheets', 'goals_conceded', 'bonus',
                'rolling_avg_points', 'points_std', 'rolling_avg_minutes',
                'rolling_avg_bonus', 'transfers_in', 'transfers_out',
                'influence', 'creativity', 'threat'
            ]
            
            # Convert numeric columns, replacing errors with NaN
            for col in numeric_columns:
                if col in df_clean.columns:
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except Exception as e:
                        print(f"Error converting {col}: {str(e)}")
                        df_clean[col] = np.nan
            
            rows_dropped = initial_rows - len(df_clean)
            
            cleaning_report = {
                'initial_rows': initial_rows,
                'final_rows': len(df_clean),
                'rows_dropped': rows_dropped,
                'rows_dropped_percentage': round((rows_dropped / initial_rows) * 100, 2)
            }
            
            return df_clean, cleaning_report


    def train_ml_model(self, model_configs,  df, gameweeks, test_set_gw):
        """Train a machine learning model for points prediction"""
        features = model_configs['features']
        model = model_configs['model']
        scaler = model_configs['scaler']
        
        # Determine the cutoff gameweek based on the most recent gameweeks
        cutoff_gameweek = gameweeks[-test_set_gw]

        # Remove rows where total_points is NaN
        df_clean = df.dropna(subset=['total_points'])
        
        # Prepare training and testing data
        train_df = df_clean[df_clean['gameweek'] < cutoff_gameweek]
        test_df = df_clean[df_clean['gameweek'] >= cutoff_gameweek]
        
        X_train = train_df[features].fillna(0)
        y_train = train_df['total_points']
        X_test = test_df[features].fillna(0)
        y_test = test_df['total_points']

        # Scale features
        if scaler is not None:
            # Fit and transform training features
            scaler.fit(X_train[features])
            X_train_scaled = scaler.transform(X_train[features])
            X_test_scaled = scaler.transform(X_test[features])
        else:
            # No scaling needed
            X_train_scaled = X_train[features].values
            X_test_scaled = X_test[features].values
        
        # Train model
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        metrics = {
            'model_name': model.__class__.__name__,
            'features': features,
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': root_mean_squared_error(y_test, predictions),
            'r2' : r2_score(y_test, predictions),
        }
        
        self.model = model
        return metrics

    def calculate_custom_expected_points(self, df, model_configs):
        """
        Calculate custom expected points for multiple models using their trained ML models.

        Args:
            df (pd.DataFrame): The dataset with player stats.
            model_configs (dict): A dictionary containing models, scalers, and features.
                                Format:
                                {
                                    'model_name': {
                                        'model': trained_model,
                                        'scaler': fitted_scaler,
                                        'features': feature_list
                                    }
                                }

        Returns:
            pd.DataFrame: The dataset with added custom expected points for each model as new columns.
        """
        if not model_configs:
            raise ValueError("No model configurations provided.")

        for model_name, config in model_configs.items():
            model = config.get('model')
            scaler = config.get('scaler')
            features = config.get('features')

            if not model or not features:
                raise ValueError(f"Incomplete configuration for model {model_name}.")

            # Ensure required columns exist and fill missing values
            df_features = df[features].fillna(0)

            # Scale features
            if scaler is not None:
                # Fit and transform training features
                scaler.fit(df_features[features])
                X_scaled = scaler.transform(df_features[features])
            else:
                # No scaling needed
                X_scaled = df_features[features].values

            # Predict custom expected points using the model
            df[f'custom_ep_{model_name}'] = pd.Series(model.predict(X_scaled)).fillna(0).astype(float)

            # Ensure values are non-negative
            df[f'custom_ep_{model_name}'] = np.maximum(df[f'custom_ep_{model_name}'], 0)

        # Remove rows where predictions couldn't be calculated
        df = df.dropna(subset=[f'custom_ep_{model_name}' for model_name in model_configs.keys()])

        return df

def main(gameweeks, test_set_gw):
    model_tester = ModelMaker()
    data = model_tester.collect_data(gameweeks)
    data = model_tester.add_features(data)
    
    # Run initial quality assessment
    initial_quality = model_tester.assess_data_quality(data)
    print("\nInitial Data Quality Report:")
    print(f"Total rows: {initial_quality['total_rows']}")
    if initial_quality['nan_analysis']:
        print("\nColumns with NaN values:")
        for col, stats in initial_quality['nan_analysis'].items():
            print(f"{col}: {stats['count']} NaN values ({stats['percentage']}%)")
    
    results = []
    for name, config in model_configs.items():
        result = model_tester.train_ml_model(config, data, gameweeks, test_set_gw)
        result['config_name'] = name
        results.append(result)

    # Calculated expected points        
    data_with_predictions = model_tester.calculate_custom_expected_points(data, model_configs)

    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results).sort_values(by='mae')

    return {
        'data': data_with_predictions,
        'results' : results_df,
        'initial_quality': initial_quality,
    }

features = [
            'now_cost', 'form', 'points_per_game', 'selected_by_percent',
            'influence', 'creativity', 'threat', 'ict_index',
            'rolling_avg_points', 'rolling_avg_minutes', 'rolling_avg_bonus', 'rolling_avg_points_3'
        ]

model_configs = {
    'linear_regression': {
        'model': LinearRegression(),
        'scaler': StandardScaler(),
        'features': features
    },
    'random_forest_v1': {
        'model': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'scaler': None,  # No need for scaling with tree-based models
        'features': features
    },
    'xgboost_v1': {
        'model': XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'scaler': StandardScaler(),
        'features': features
    },
    'lightgbm_v1': {
        'model': LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'scaler': None,  # LightGBM can handle unscaled features
        'features': features
    },
    'gradient_boosting_v1': {
        'model': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'scaler': None,  # Gradient boosting handles unscaled features
        'features': features
    },
    'decision_tree_v1': {
        'model': DecisionTreeRegressor(max_depth=10, random_state=42),
        'scaler': None,  # Tree-based models do not need scaling
        'features': features
    },
    'ridge_regression_v1': {
        'model': Ridge(alpha=1.0),
        'scaler': StandardScaler(),
        'features': features
    },
    'lasso_regression_v1': {
        'model': Lasso(alpha=0.1),
        'scaler': StandardScaler(),
        'features': features
    },
    'lasso_regression_v2': {
        'model': Lasso(alpha=0.01),
        'scaler': StandardScaler(),
        'features': features
    },
    'svr_v1': {
        'model': SVR(kernel='rbf', C=100, gamma=0.1),
        'scaler': StandardScaler(),  # SVR requires feature scaling
        'features': features
    },
    'knn_v1': {
        'model': KNeighborsRegressor(n_neighbors=10),
        'scaler': MinMaxScaler(),  # KNN is distance-based, so scaling is essential
        'features': features
    },
    'mlp_v1': {
        'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'scaler': StandardScaler(),  # Neural networks perform better with scaled inputs
        'features': features
    },
    'catboost_model': {
        'model': CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=0),
        'scaler': None,  # CatBoost can handle unscaled features
        'features': features
    }
}

if __name__ == "__main__":
    gameweeks = list(range(1, 23))  # Select gameweek range
    test_set_gw = 4 # Use as test set
    results = main(gameweeks, test_set_gw)
    print(results['results'][['config_name', 'mae', 'rmse', 'r2']])

    # print(results['data'].head(5))
    # results['results'].plot(x='config_name', y=['mae', 'rmse'], kind='bar', title='Model Performance')
    # plt.show()

# analyze_results(results)