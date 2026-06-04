import numpy as np

#scales numeric data so the mean becomes 0 and standard deviation becomes 1
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.var = None
        self.nseen = 0

    #calculates mean and std from the data
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean = np.nanmean(X, axis=0)
        self.var = np.nanvar(X, axis=0)
        self.nseen = X.shape[0]
        return self

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)

        cmean = np.nanmean(X, axis=0)
        cvar  = np.nanvar(X, axis=0)
        cn    = X.shape[0]

        if self.nseen == 0:
            self.mean  = cmean
            self.var   = cvar
            self.nseen = cn
            return self

        total_n = self.nseen + cn
        delta   = cmean - self.mean  # difference between the two means

        new_mean = (self.mean * self.nseen + cmean * cn) / total_n

        # Parallel variance formula (Chan et al.)
        # The extra term accounts for the variance introduced by the mean shift
        new_var = (
            self.var   * self.nseen +
            cvar       * cn +
            delta**2   * self.nseen * cn / total_n   # ← this was missing
        ) / total_n

        self.mean  = new_mean
        self.var   = new_var
        self.nseen = total_n

        return self

    #applies scaling using the stored mean and var
    def transform(self, X):
        if self.mean is None or self.var is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        
        std = np.sqrt(self.var)
        std[std == 0] = 1

        return (X - self.mean) / std 
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
#scales data to a fixed range between 0 and 1
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    #finds minimum and maximum values from the data
    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    def partial_fit(self, X):
        cmin = np.min(X, axis=0)
        cmax = np.max(X, axis=0)

        if self.min is None or self.max is None:
            self.min = cmin
            self.max = cmax
            return self

        self.min = np.minimum(self.min, cmin)
        self.max = np.maximum(self.max, cmax)

        return self

    #scales values using stored min and max
    def transform(self, X):
        if self.min is None or self.max is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        d = self.max - self.min
        d[d == 0] = 1

        return (X - self.min) / d

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

#converts categorical values into one-hot encoder vectors
class OneHotEncoder:
    def __init__(self):
        self.categories = None

    #finds unique categories for each column
    def fit(self, X):
        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def partial_fit(self, X):
        c_categories = [np.unique(X[:, i]) for i in range(X.shape[1])]

        if self.categories is None:
            self.categories = c_categories
            return self

        for i in range(len(self.categories)):
            self.categories[i] = np.unique(np.concatenate([self.categories[i], c_categories[i]]))

        return self

    #creates one-hot encoded columns based on learned categories
    def transform(self, X):
        if self.categories is None:
            raise ValueError("OneHotEncoder has not been fitted yet.")
        encoded_columns = []

        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories[i]

            one_hot = (col[:, None] == categories).astype(int)
            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

#applies different preprocessing to numeric and categorical columns
class ColumnTransformer:
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    #fits scaler on numeric columns and encoder on categorical columns
    def fit(self, X):
        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            self.scaler.fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.fit(X_cat)
        
        return self
    
    def partial_fit(self, X):
        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            self.scaler.partial_fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.partial_fit(X_cat)
        
        return self

    #transforms numeric and categorical parts separately and combines them
    def transform(self, X):
        if (len(self.num_cols) > 0 and self.scaler.mean is None) or \
           (len(self.cat_cols) > 0 and self.encoder.categories is None):
            raise ValueError("ColumnTransformer has not been fitted yet.")
        
        parts = []

        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            parts.append(self.scaler.transform(X_num))

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            parts.append(self.encoder.transform(X_cat))

        if len(parts) == 0:
            raise ValueError("No columns specified.")

        return np.hstack(parts)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class SimpleImputer:
    
    def __init__(self):
        self.fill_values = None
        self.n_samples_seen = 0

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.fill_values = np.nanmean(X, axis=0)
        self.n_samples_seen = X.shape[0]

        return self

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)

        chunk_mean = np.nanmean(X, axis=0)
        chunk_n = X.shape[0]

        if self.fill_values is None:
            self.fill_values = chunk_mean
            self.n_samples_seen = chunk_n
            return self

        total_n = self.n_samples_seen + chunk_n

        self.fill_values = (
            self.fill_values * self.n_samples_seen +
            chunk_mean * chunk_n
        ) / total_n

        self.n_samples_seen = total_n

        return self

    def transform(self, X):
        if self.fill_values is None:
            raise ValueError("SimpleImputer has not been fitted yet.")

        X = np.array(X, dtype=float, copy=True)

        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.fill_values, inds[1])

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
