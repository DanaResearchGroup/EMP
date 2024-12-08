from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mmltoolkit.featurizations import sum_over_bonds, Estate_CDS_SoB_featurizer

input_path = r"C:\Users\samco\OneDrive\Desktop\school\Work in Alon's lab\EMP\datasets\combined_data.xlsx"
df = pd.read_excel(input_path)

smiles_column = 'SMILES'
target_columns = ["heat of explosion (KJ/Kg)"]

# Extract SMILES and target values
smis = df.loc[:306, smiles_column].values
ys = df.loc[:306, target_columns].values

mol_list = [Chem.AddHs(Chem.MolFromSmiles(smile)) for smile in smis]
names_Estate_CDS_SoB, X_Estate_CDS_SoB = Estate_CDS_SoB_featurizer(mol_list)
bond_types, X_LBoB  = sum_over_bonds(mol_list)

def smiles_to_fingerprint(smiles, fp_size=2048):
    try:
        # Ensure SMILES is a string
        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
            return np.array(fp)  # Convert to NumPy array for compatibility
        else:
            return None  # Return None for invalid SMILES
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

best_featurizations= {}
results = {}

for feat_name, X_feat in featurizations.items():
    print(f"Optimizing for featurization: {feat_name}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, ys, test_size=0.2, random_state=1
    )

    # Hyperparameter optimization
    model = xgb.XGBRegressor()
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
    search = RandomizedSearchCV(model, param_grid, n_iter=50, scoring='neg_mean_absolute_error', cv=5, verbose=1,
                                random_state=1, n_jobs=-1)
    search.fit(X_train, y_train)

    # Save the best model and metrics
    best_featurization = search.best_estimator_
    best_featurizations[feat_name] = best_featurization

    # Evaluate the model
    predictions = best_featurization.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[feat_name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
    print(f"Results for {feat_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

print("\nSummary of Results:")
for feat_name, metrics in results.items():
    print(f"{feat_name}: MAE = {metrics['MAE']:.3f}, RMSE = {metrics['RMSE']:.3f}, R² = {metrics['R²']:.3f}")
