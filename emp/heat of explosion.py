import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mmltoolkit.featurizations import sum_over_bonds, Estate_CDS_SoB_featurizer

input_path = r"/home/sam-cogan/Code/EMP/datasets/combined_data.xlsx"
df = pd.read_excel(input_path)

smiles_column = 'SMILES'
target_columns = ["heat of explosion (KJ/Kg)"]

# Extract SMILES and target values
smis = df.loc[:306, smiles_column].values
ys = df.loc[:306, target_columns].values

mol_list = [Chem.AddHs(Chem.MolFromSmiles(smile)) for smile in smis]
names_Estate_CDS_SoB, X_Estate_CDS_SoB = Estate_CDS_SoB_featurizer(mol_list)
bond_types, X_LBoB = sum_over_bonds(mol_list)


def smiles_to_fingerprint(smiles, fp_size=2048):
    try:
        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
            return np.array(fp)
        else:
            return None
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


# Convert all SMILES to fingerprints
fingerprints = [smiles_to_fingerprint(smi) for smi in smis]

featurizations = {
    "X_Estate_CDS_SoB": X_Estate_CDS_SoB,
    "X_LBoB": X_LBoB,
    "fingerprints": fingerprints
}

models = {
    "XGBRegressor": xgb.XGBRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "SVR": SVR(),
    "LinearRegression": LinearRegression(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor()
}

results = {}
test_outputs = {}

for feat_name, X_feat in featurizations.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, ys, test_size=0.2, random_state=1
    )

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    for model_name, model in models.items():
        print(f"Training {model_name} on {feat_name}")

        if model_name == "XGBRegressor":
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 1, 5],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 2, 5],
            }
        else:
            param_grid = {'n_estimators': [10, 50, 100]} if hasattr(model, 'n_estimators') else {}

        if param_grid:
            search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_absolute_error', cv=5,
                                        verbose=1, random_state=1, n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_model = model

        predictions = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results[(feat_name, model_name)] = {"MAE": mae, "RMSE": rmse, "R²": r2}

        # Save y_test and predictions for later analysis
        test_outputs[(feat_name, model_name)] = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions
        })

        print(f"Results for {feat_name} with {model_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

print("\nSummary of Results:")
for (feat_name, model_name), metrics in results.items():
    print(
        f"{feat_name} with {model_name}: MAE = {metrics['MAE']:.3f}, RMSE = {metrics['RMSE']:.3f}, R² = {metrics['R²']:.3f}")
