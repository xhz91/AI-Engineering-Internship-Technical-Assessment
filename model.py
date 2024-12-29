import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Union

class MerchantPricingModel:
    """
    A model to assess whether a merchant's card processing fees are competitive,
    neutral, or non-competitive based on market data and merchant characteristics.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fee_thresholds = {
            'competitive': 0.004804,  # 33rd percentile
            'neutral': 0.008360  # 66th percentile
        }
        
    def _parse_fee(self, fee_str: str) -> Dict[str, float]:
        """Parse a fee string into percentage and fixed components."""
        try:
            percent_str, pence_str = fee_str.split('+')
            percent = float(percent_str.replace('%', '').strip()) / 100
            pence = float(pence_str.replace('p', '').strip()) / 100
            return {'percent': percent, 'fixed': pence}
        except:
            return {'percent': 0, 'fixed': 0}
    
    def _calculate_weighted_fee(self, row: pd.Series) -> float:
        """Calculate weighted average fee based on card type distribution."""
        
        # Parse all fees
        fees = {
            'mc_debit': self._parse_fee(row['Mastercard Debit']),
            'visa_debit': self._parse_fee(row['Visa Debit']),
            'mc_credit': self._parse_fee(row['Mastercard Credit']),
            'visa_credit': self._parse_fee(row['Visa Credit']),
            'mc_biz_debit': self._parse_fee(row['Mastercard Business Debit']),
            'visa_biz_debit': self._parse_fee(row['Visa Business Debit'])
        }
        
        # Calculate weighted average based on assumptions:
        # 40% Mastercard, 60% Visa
        # 90% debit, 8% credit, 2% business debit
        weighted_percent = (
            # Debit cards (90%)
            0.9 * (
                0.4 * fees['mc_debit']['percent'] +
                0.6 * fees['visa_debit']['percent']
            ) +
            # Credit cards (8%)
            0.08 * (
                0.4 * fees['mc_credit']['percent'] +
                0.6 * fees['visa_credit']['percent']
            ) +
            # Business debit cards (2%)
            0.02 * (
                0.4 * fees['mc_biz_debit']['percent'] +
                0.6 * fees['visa_biz_debit']['percent']
            )
        )
        
        return weighted_percent
    
    def _prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Prepare features for the model."""
        # Calculate weighted fee for each merchant
        df['weighted_fee'] = df.apply(self._calculate_weighted_fee, axis=1)
        
        # Create price competitiveness labels
        df['price_category'] = pd.cut(
            df['weighted_fee'],
            bins=[-np.inf, self.fee_thresholds['competitive'], 
                  self.fee_thresholds['neutral'], np.inf],
            labels=['competitive', 'neutral', 'non_competitive']
        )
        
        # Prepare binary features
        df['Is Registered'] = df['Is Registered'].map({'Yes': 1, 'No': 0})
        df['Accepts Card'] = df['Accepts Card'].map({'Yes': 1, 'No': 0})

        # Prepare categorical features
        categorical_columns = ['MCC Code', 'Current Provider']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Select features for the model
        feature_columns = [
            'Annual Card Turnover',
            'Average Transaction Amount',
            'MCC Code_encoded',
            'Is Registered',
            'Accepts Card',
            'Current Provider_encoded'
        ]

        # Apply scaling
        if fit_scaler:
            scaled_features = self.scaler.fit_transform(df[feature_columns])
        else:
            scaled_features = self.scaler.transform(df[feature_columns])

        scaled_feature_columns = pd.DataFrame(scaled_features, columns=feature_columns)
        return scaled_feature_columns
    

    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data, then test the model."""
        X = self._prepare_features(df, fit_scaler=True)
        y = pd.cut(
            df.apply(self._calculate_weighted_fee, axis=1),
            bins=[-np.inf, self.fee_thresholds['competitive'], 
                  self.fee_thresholds['neutral'], np.inf],
            labels=['competitive', 'neutral', 'non_competitive']
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)
        
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print(f"accuracy score: {accuracy_score(y_test, y_pred)}")

        return None
    

    def predict(self, merchant_data: Union[pd.DataFrame, Dict]) -> str:
        """
        Predict whether a merchant's pricing is competitive, neutral, or non-competitive.
        
        Args:
            merchant_data: DataFrame or dictionary containing merchant information
            
        Returns:
            str: 'competitive', 'neutral', or 'non_competitive'
        """
        if isinstance(merchant_data, dict):
            merchant_data = pd.DataFrame([merchant_data])
            
        X = self._prepare_features(merchant_data, fit_scaler=False)
        return self.model.predict(X)[0]


# Example usage
if __name__ == "__main__":
    model = MerchantPricingModel()
    df = pd.read_csv('data.csv', encoding='ISO-8859-1')
    model.fit(df)

    # Make a prediction for a new merchant
    new_merchant = {
        'Annual Card Turnover': 10000000,
        'Average Transaction Amount': 1000,
        'MCC Code': 742,
        'Is Registered': 'Yes',
        'Accepts Card': 'Yes',
        'Current Provider': 'stripe',
        'Mastercard Debit': '0.1% + 2p',
        'Visa Debit': '0.1% + 2p',
        'Mastercard Credit': '0.1% + 2p',
        'Visa Credit': '0.1% + 2p',
        'Mastercard Business Debit': '0.1% + 2p',
        'Visa Business Debit': '0.1% + 2p'
    }
    
    result = model.predict(new_merchant)
    print(f"For the example new business, the pricing competitiveness is: {result}")
    
    