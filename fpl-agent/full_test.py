import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import optuna
from optuna.integration import LightGBMPruningCallback
import shap

# ----- 1. DATA PREPARATION -----

def load_and_prepare_data(file_path):
    """Load and prepare the FPL data for modeling."""
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime if they exist
    if 'window_start' in df.columns and 'features_end' in df.columns:
        df['window_start'] = pd.to_datetime(df['window_start'])
        df['features_end'] = pd.to_datetime(df['features_end'])
    
    # Sort by gameweek and player
    df = df.sort_values(['target_gameweek', 'player_id'])
    
    return df

def add_fixture_features(df, fixtures_df):
    """Add fixture difficulty and home/away features.
    This assumes you have a separate fixtures dataframe."""
    
    # Join fixture data to player data
    df = df.merge(
        fixtures_df[['gameweek', 'team', 'is_home', 'opponent_team', 'difficulty']],
        left_on=['target_gameweek', 'team'],
        right_on=['gameweek', 'team'],
        how='left'
    )
    
    # Create features for fixture difficulty
    df['is_home'] = df['is_home'].astype(int)
    
    return df

def feature_selection(df, position=None):
    """Select relevant features based on position."""
    
    # Filter by position if specified
    if position:
        df = df[df['position'] == position]
    
    # Drop identifier columns for modeling
    id_cols = ['player_id', 'player_name', 'team', 'window_start', 'features_end']
    
    # Separate target from features
    X = df.drop(['actual_points'] + id_cols, axis=1)
    y = df['actual_points']
    
    # Store position and gameweek for later analysis
    meta_cols = ['position', 'target_gameweek']
    meta = df[meta_cols + id_cols] if all(col in df.columns for col in meta_cols) else df[id_cols]
    
    return X, y, meta

def reduce_dimensionality(X, threshold=0.95):
    """Reduce feature set by removing highly correlated features."""
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Create upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation higher than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop highly correlated features
    X_reduced = X.drop(to_drop, axis=1)
    
    print(f"Reduced features from {X.shape[1]} to {X_reduced.shape[1]}")
    
    return X_reduced


# ----- 2. POSITION-SPECIFIC MODELING -----

def train_position_models(df, positions=['GK', 'DEF', 'MID', 'FWD'], cv_splits=5):
    """Train separate models for each position."""
    
    position_models = {}
    position_importances = {}
    position_metrics = {}
    
    for position in positions:
        print(f"\nTraining model for {position} position...")
        
        # Get position-specific data
        pos_df = df[df['position'] == position].copy()
        
        if len(pos_df) < 100:  # Skip if not enough data
            print(f"Not enough data for {position}, skipping.")
            continue
            
        # Prepare features
        X, y, meta = feature_selection(pos_df)
        X = reduce_dimensionality(X, threshold=0.9)  # More aggressive for position-specific
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Store evaluation metrics
        mae_scores = []
        rmse_scores = []
        
        # Setup to store feature importances
        feature_importances = pd.DataFrame(index=X.columns)
        
        # Training and validation
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model with hyperparameters from optimization
            model = LGBMRegressor(
                n_estimators=500, 
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            
            # Store feature importances
            importances = pd.Series(model.feature_importances_, index=X.columns)
            feature_importances = pd.concat([feature_importances, importances], axis=1)
        
        # Train final model on all data
        final_model = LGBMRegressor(
            n_estimators=500, 
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=42
        )
        
        final_model.fit(X_scaled, y)
        
        # Store results
        position_models[position] = {
            'model': final_model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        position_metrics[position] = {
            'mae': np.mean(mae_scores),
            'rmse': np.mean(rmse_scores)
        }
        
        position_importances[position] = feature_importances.mean(axis=1).sort_values(ascending=False)
        
        print(f"{position} model - Mean MAE: {np.mean(mae_scores):.3f}, Mean RMSE: {np.mean(rmse_scores):.3f}")
    
    return position_models, position_importances, position_metrics


# ----- 3. HYPERPARAMETER OPTIMIZATION -----

def optimize_hyperparams(X, y, n_trials=50):
    """Use Optuna to optimize LightGBM hyperparameters."""
    
    # Define the objective function
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = LGBMRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores.append(mae)
        
        return np.mean(scores)
    
    # Create and optimize study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters:", study.best_params)
    return study.best_params


# ----- 4. FEATURE IMPORTANCE ANALYSIS -----

def analyze_feature_importance(model, X, position=None):
    """Analyze feature importance using SHAP values."""
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 10))
    title = f"Feature Importance for {position}" if position else "Feature Importance"
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    
    # Plot SHAP dependencies for top features
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-mean_shap_values)[:10]
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_indices):
        feature_name = X.columns[idx]
        shap.dependence_plot(
            idx, shap_values, X, 
            ax=axes[i],
            show=False
        )
        axes[i].set_title(f"SHAP Dependence: {feature_name}")
    
    plt.tight_layout()
    
    # Return important features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_shap_values
    }).sort_values('importance', ascending=False)
    
    return feature_importance


# ----- 5. PREDICTION FOR NEW GAMEWEEKS -----

def predict_next_gameweek(models, new_data, gameweek):
    """Predict points for the next gameweek."""
    
    # Process each player based on their position
    predictions = []
    
    for position in models.keys():
        # Filter players by position
        pos_players = new_data[new_data['position'] == position].copy()
        
        if len(pos_players) == 0:
            continue
            
        # Get model components
        model = models[position]['model']
        scaler = models[position]['scaler']
        features = models[position]['features']
        
        # Prepare features
        X = pos_players[features].copy()
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        
        # Predict
        pos_players['predicted_points'] = model.predict(X_scaled)
        
        # Add to results
        predictions.append(pos_players[['player_id', 'player_name', 'team', 'position', 'predicted_points']])
    
    # Combine all predictions
    all_predictions = pd.concat(predictions)
    all_predictions['gameweek'] = gameweek
    
    return all_predictions.sort_values('predicted_points', ascending=False)


# ----- 6. ENSEMBLE MODELING -----

def train_ensemble_model(df, cv_splits=5):
    """Train ensemble model combining base models and meta features."""
    
    # First level: Train position-specific models
    position_models, _, _ = train_position_models(df, cv_splits=cv_splits)
    
    # Second level: Train stacking model
    meta_features = []
    
    # Use time series cross-validation to generate meta-features
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Record original order to reconstruct dataframe
    df['original_index'] = np.arange(len(df))
    
    for train_idx, val_idx in tscv.split(df):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        
        # Train position models on training data
        train_pos_models, _, _ = train_position_models(train_df, cv_splits=3)
        
        # Generate predictions for validation set
        val_predictions = []
        
        for position, model_info in train_pos_models.items():
            pos_val = val_df[val_df['position'] == position]
            
            if len(pos_val) == 0:
                continue
                
            # Prepare features for validation
            X, _, meta = feature_selection(pos_val)
            X = X[model_info['features']]  # Keep only model's features
            X_scaled = pd.DataFrame(
                model_info['scaler'].transform(X), 
                columns=X.columns
            )
            
            # Make predictions
            pos_val = meta.copy()
            pos_val['position_model_pred'] = model_info['model'].predict(X_scaled)
            val_predictions.append(pos_val)
        
        # Combine predictions from all positions
        if val_predictions:
            val_meta = pd.concat(val_predictions)
            val_meta['original_index'] = val_df['original_index'].values
            meta_features.append(val_meta)
    
    # Combine all meta-features
    meta_df = pd.concat(meta_features)
    meta_df = meta_df.sort_values('original_index')
    
    # Join with original data
    df = df.merge(
        meta_df[['original_index', 'position_model_pred']], 
        on='original_index', how='left'
    )
    
    # Create meta-model features
    meta_X, meta_y, _ = feature_selection(df)
    
    # Add position-specific predictions as a feature
    meta_X['position_model_pred'] = df['position_model_pred']
    
    # Train final ensemble model
    final_model = LGBMRegressor(
        n_estimators=500, 
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42
    )
    
    # Scale features
    meta_scaler = StandardScaler()
    meta_X_scaled = pd.DataFrame(
        meta_scaler.fit_transform(meta_X), 
        columns=meta_X.columns
    )
    
    # Train model
    final_model.fit(meta_X_scaled, meta_y)
    
    # Return models and preprocessing components
    return {
        'position_models': position_models,
        'meta_model': final_model,
        'meta_scaler': meta_scaler,
        'meta_features': meta_X.columns.tolist()
    }


# ----- 7. MODEL EVALUATION -----

def evaluate_model(ensemble, test_df):
    """Evaluate ensemble model on test data."""
    
    # Generate position-specific predictions
    test_df['position_model_pred'] = np.nan
    
    for position, model_info in ensemble['position_models'].items():
        pos_test = test_df[test_df['position'] == position]
        
        if len(pos_test) == 0:
            continue
            
        # Prepare features
        X, _, _ = feature_selection(pos_test)
        X = X[model_info['features']]  # Keep only model's features
        X_scaled = pd.DataFrame(
            model_info['scaler'].transform(X), 
            columns=X.columns
        )
        
        # Make predictions
        test_df.loc[test_df['position'] == position, 'position_model_pred'] = model_info['model'].predict(X_scaled)
    
    # Prepare meta features
    meta_X, meta_y, _ = feature_selection(test_df)
    meta_X['position_model_pred'] = test_df['position_model_pred']
    
    # Keep only meta features used in model
    meta_X = meta_X[ensemble['meta_features']]
    
    # Scale features
    meta_X_scaled = pd.DataFrame(
        ensemble['meta_scaler'].transform(meta_X), 
        columns=meta_X.columns
    )
    
    # Make predictions
    test_df['predicted_points'] = ensemble['meta_model'].predict(meta_X_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(test_df['actual_points'], test_df['predicted_points'])
    rmse = np.sqrt(mean_squared_error(test_df['actual_points'], test_df['predicted_points']))
    
    # Calculate position-specific metrics
    position_metrics = {}
    
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_df = test_df[test_df['position'] == position]
        
        if len(pos_df) == 0:
            continue
            
        pos_mae = mean_absolute_error(pos_df['actual_points'], pos_df['predicted_points'])
        pos_rmse = np.sqrt(mean_squared_error(pos_df['actual_points'], pos_df['predicted_points']))
        
        position_metrics[position] = {
            'mae': pos_mae,
            'rmse': pos_rmse,
            'count': len(pos_df)
        }
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(test_df['actual_points'], test_df['predicted_points'], alpha=0.5)
    plt.plot([0, test_df['actual_points'].max()], [0, test_df['actual_points'].max()], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs Predicted FPL Points')
    plt.grid(True)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    errors = test_df['predicted_points'] - test_df['actual_points']
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    
    # Return results
    results = {
        'overall': {
            'mae': mae,
            'rmse': rmse
        },
        'position_metrics': position_metrics,
        'predictions': test_df[['player_id', 'player_name', 'team', 'position', 'actual_points', 'predicted_points']]
    }
    
    return results


# ----- 8. MAIN WORKFLOW -----

def main():
    """Main workflow for FPL points prediction."""
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data('fpl_player_data.csv')
    
    # Split data by gameweek for proper time series validation
    train_gws = range(1, 10)  # First 25 gameweeks for training
    test_gws = range(11, 15)  # Gameweeks 26-30 for testing
    
    train_df = df[df['target_gameweek'].isin(train_gws)].copy()
    test_df = df[df['target_gameweek'].isin(test_gws)].copy()
    
    print(f"Training data: {len(train_df)} rows, {train_df['target_gameweek'].nunique()} gameweeks")
    print(f"Test data: {len(test_df)} rows, {test_df['target_gameweek'].nunique()} gameweeks")
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    ensemble = train_ensemble_model(train_df, cv_splits=5)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    results = evaluate_model(ensemble, test_df)
    
    # Print metrics
    print("\nOverall metrics:")
    print(f"MAE: {results['overall']['mae']:.3f}")
    print(f"RMSE: {results['overall']['rmse']:.3f}")
    
    print("\nPosition-specific metrics:")
    for position, metrics in results['position_metrics'].items():
        print(f"{position}: MAE = {metrics['mae']:.3f}, RMSE = {metrics['rmse']:.3f}, n = {metrics['count']}")
    
    # Make predictions for next gameweek
    print("\nPredicting next gameweek...")
    next_gw = max(df['target_gameweek']) + 1
    next_gw_data = prepare_next_gameweek_data(df, next_gw)
    
    predictions = predict_next_gameweek(ensemble['position_models'], next_gw_data, next_gw)
    print(f"Top 10 predicted players for GW{next_gw}:")
    print(predictions.head(10))
    
    return ensemble, results


def prepare_next_gameweek_data(df, next_gw):
    """Prepare data for the next gameweek prediction."""
    
    # Get the most recent data for each player
    latest_df = df[df['target_gameweek'] == max(df['target_gameweek'])].copy()
    
    # Update for next gameweek
    latest_df['target_gameweek'] = next_gw
    
    # Here we would normally update features for the new gameweek
    # This would include:
    # 1. Rolling window features (already in the data)
    # 2. Fixture information for the next gameweek (would need fixture data)
    # 3. Any other time-dependent features
    
    return latest_df


if __name__ == "__main__":
    ensemble, results = main()