import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class FPLAnalyzer:
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


    def train_ml_model(self, df, gameweeks, test_set_gw):
        """Train a machine learning model for points prediction"""
        features = [
            'now_cost', 'form', 'points_per_game', 'selected_by_percent',
            'influence', 'creativity', 'threat', 'ict_index',
            'rolling_avg_points', 'rolling_avg_minutes', 'rolling_avg_bonus'
        ]
        
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
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Calculate prediction metrics
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
        
        self.model = model
        return metrics

    def calculate_custom_expected_points(self, df):
        """
        Calculate custom expected points using the trained ML model and apply the same scaling.

        Args:
            df (pd.DataFrame): The dataset with player stats.
            model (sklearn model): The trained machine learning model.
            scaler (sklearn scaler): The fitted scaler used in training.
            features (list): The feature columns used in training.

        Returns:
            pd.Series: Predicted expected points.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Please train the model first.")

        features = [
            'now_cost', 'form', 'points_per_game', 'selected_by_percent',
            'influence', 'creativity', 'threat', 'ict_index',
            'rolling_avg_points', 'rolling_avg_minutes', 'rolling_avg_bonus'
        ]
        # Ensure required columns exist and fill missing values
        df_features = df[features].fillna(0)

        # Apply the same scaler from training
        X_scaled = self.scaler.transform(df_features)

        # Predict custom expected points using the trained model
        df['custom_ep'] = pd.Series(self.model.predict(X_scaled)).fillna(0).astype(float)

        # Ensure values are non-negative (as negative expected points don't make sense)
        df['custom_ep'] = np.maximum(df['custom_ep'], 0)
        
        # Remove rows where total_points is NaN
        df = df.dropna(subset=['custom_ep'])

        return df

    def analyze_prediction_accuracy(self, df,  gameweeks, test_set_gw):
            """Analyze prediction accuracy with detailed breakdowns"""
            # Clean data first
            df_clean, cleaning_report = self.clean_data_for_analysis(df)
            
            # Run quality assessment
            quality_report = self.assess_data_quality(df_clean)

            # Determine the cutoff gameweek based on the most recent gameweeks
            cutoff_gameweek = gameweeks[-test_set_gw]
            
            # Prepare training and testing data
            df_clean = df_clean[df_clean['gameweek'] >= cutoff_gameweek] 
            
            # Ensure numeric columns before calculations
            df_clean['total_points'] = pd.to_numeric(df_clean['total_points'], errors='coerce')
            df_clean['ep_next'] = pd.to_numeric(df_clean['ep_next'], errors='coerce')
            df_clean['custom_ep'] = pd.to_numeric(df_clean['custom_ep'], errors='coerce')
            
            # Drop any remaining NaN after conversion
            df_clean = df_clean.dropna(subset=['total_points', 'ep_next', 'custom_ep'])
            
            metrics = {
                'official_ep': {
                    'mae': mean_absolute_error(df_clean['total_points'], df_clean['ep_next']),
                    'rmse': np.sqrt(mean_squared_error(df_clean['total_points'], df_clean['ep_next']))
                },
                'custom_ep': {
                    'mae': mean_absolute_error(df_clean['total_points'], df_clean['custom_ep']),
                    'rmse': np.sqrt(mean_squared_error(df_clean['total_points'], df_clean['custom_ep']))
                }
            }
            
            # Analysis by position with explicit type handling
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            df_clean['position_label'] = df_clean['element_type'].map(position_map)
            position_analysis = df_clean.groupby('position_label').agg({
                'ep_next': ['mean', 'std'],
                'custom_ep': ['mean', 'std'],
                'total_points': ['mean', 'std']
            }).round(2)  # Round to avoid floating point issues

            # Analysis by price range with bin values displayed
            price_bins, bin_labels = pd.qcut(df_clean['now_cost'], 4, retbins=True, labels=['Budget', 'Mid-Low', 'Mid-High', 'Premium'])
            df_clean['price_range'] = price_bins
            price_analysis = df_clean.groupby('price_range').agg({
                'ep_next': ['mean', 'std'],
                'custom_ep': ['mean', 'std'],
                'total_points': ['mean', 'std']
            }).round(2)  # Round to avoid floating point issues

            # Displaying the price bin ranges for reference
            price_ranges_info = dict(zip(bin_labels, bin_labels[1:]))
            
            return {
                'metrics': metrics,
                'position_analysis': position_analysis,
                'price_ranges' : price_ranges_info,
                'price_analysis': price_analysis,
                'data_quality': quality_report,
                'cleaning_report': cleaning_report
            }

def main(gameweeks, test_set_gw):
    analyzer = FPLAnalyzer()
    data = analyzer.collect_data(gameweeks)
    
    # Run initial quality assessment
    initial_quality = analyzer.assess_data_quality(data)
    print("\nInitial Data Quality Report:")
    print(f"Total rows: {initial_quality['total_rows']}")
    if initial_quality['nan_analysis']:
        print("\nColumns with NaN values:")
        for col, stats in initial_quality['nan_analysis'].items():
            print(f"{col}: {stats['count']} NaN values ({stats['percentage']}%)")
    
    # Clean and process data
    ml_metrics = analyzer.train_ml_model(data, gameweeks, test_set_gw)
    data = analyzer.calculate_custom_expected_points(data)

    accuracy_analysis = analyzer.analyze_prediction_accuracy(data, gameweeks, test_set_gw)
    
    print("\nData Cleaning Report:")
    cleaning_report = accuracy_analysis['cleaning_report']
    print(f"Initial rows: {cleaning_report['initial_rows']}")
    print(f"Rows dropped: {cleaning_report['rows_dropped']} ({cleaning_report['rows_dropped_percentage']}%)")
    print(f"Final rows: {cleaning_report['final_rows']}")
    
    return {
        'ml_metrics': ml_metrics,
        'accuracy_analysis': accuracy_analysis,
        'initial_quality': initial_quality,
        'data': data
    }

def analyze_results(results):
    """Analyze and display results from the main function in a structured way"""
    print("\n" + "="*50)
    print("DATA QUALITY ASSESSMENT")
    print("="*50)
    
    # Initial data quality
    initial_quality = results['initial_quality']
    print(f"\nInitial Dataset Size: {initial_quality['total_rows']} rows")
    
    if initial_quality['nan_analysis']:
        print("\nNaN Values Found:")
        print("-"*30)
        for col, stats in initial_quality['nan_analysis'].items():
            print(f"{col:25} {stats['count']:6} rows ({stats['percentage']:5.1f}%)")
    
    # ML Model Performance
    print("\n" + "="*50)
    print("ML MODEL PERFORMANCE")
    print("="*50)
    
    ml_metrics = results['ml_metrics']
    print(f"\nTraining Set Performance:")
    print(f"MAE: {ml_metrics['train_mae']:.3f}")
    print(f"RMSE: {ml_metrics['train_rmse']:.3f}")
    
    print(f"\nTest Set Performance:")
    print(f"MAE: {ml_metrics['test_mae']:.3f}")
    print(f"RMSE: {ml_metrics['test_rmse']:.3f}")
    
    print("\nTop 5 Most Important Features:")
    sorted_features = sorted(
        ml_metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, importance in sorted_features:
        print(f"{feature:25} {importance:.3f}")
    
    # Prediction Accuracy Analysis
    print("\n" + "="*50)
    print("PREDICTION ACCURACY COMPARISON")
    print("="*50)
    
    accuracy = results['accuracy_analysis']['metrics']
    print("\nOfficial Expected Points:")
    print(f"MAE: {accuracy['official_ep']['mae']:.3f}")
    print(f"RMSE: {accuracy['official_ep']['rmse']:.3f}")
    
    print("\nCustom Expected Points:")
    print(f"MAE: {accuracy['custom_ep']['mae']:.3f}")
    print(f"RMSE: {accuracy['custom_ep']['rmse']:.3f}")
    
    # Position Analysis
    print("\n" + "="*50)
    print("POSITION-BASED ANALYSIS")
    print("="*50)
    
    position_analysis = results['accuracy_analysis']['position_analysis']
    for position_type in position_analysis.index:
        print(f"\nPosition {position_type}:")
        print(f"Official EP Mean: {position_analysis.loc[position_type, ('ep_next', 'mean')]:.2f}")
        print(f"Custom EP Mean: {position_analysis.loc[position_type, ('custom_ep', 'mean')]:.2f}")
        print(f"Actual Points Mean: {position_analysis.loc[position_type, ('total_points', 'mean')]:.2f}")
    
    # Price Analysis
    print("\n" + "="*50)
    print("PRICE-BASED ANALYSIS")
    print("="*50)
    
    price_ranges = results['accuracy_analysis']['price_ranges']

    print("\nPrice Ranges:")
    for low, high in price_ranges.items():
        print(f"£{float(low)/10:.1f}m - £{float(high)/10:.1f}m")

    price_analysis = results['accuracy_analysis']['price_analysis']
    for price_range in price_analysis.index:
        print(f"\nPrice Range {price_range}:")
        print(f"Official EP Mean: {price_analysis.loc[price_range, ('ep_next', 'mean')]:.2f}")
        print(f"Custom EP Mean: {price_analysis.loc[price_range, ('custom_ep', 'mean')]:.2f}")
        print(f"Actual Points Mean: {price_analysis.loc[price_range, ('total_points', 'mean')]:.2f}")
    
    # Data Cleaning Summary
    print("\n" + "="*50)
    print("DATA CLEANING SUMMARY")
    print("="*50)
    
    cleaning_report = results['accuracy_analysis']['cleaning_report']
    print(f"\nInitial Rows: {cleaning_report['initial_rows']}")
    print(f"Rows Dropped: {cleaning_report['rows_dropped']} ({cleaning_report['rows_dropped_percentage']:.1f}%)")
    print(f"Final Rows: {cleaning_report['final_rows']}")

if __name__ == "__main__":
    gameweeks = list(range(1, 20))  # Select gameweek range
    test_set_gw = 4 # Use as test set
    results = main(gameweeks, test_set_gw)
    analyze_results(results)