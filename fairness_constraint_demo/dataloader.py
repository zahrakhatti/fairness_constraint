import torch
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSTravelTime, ACSPublicCoverage, ACSMobility
from sklearn.preprocessing import RobustScaler
from config import get_args


args = get_args()

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use relative paths for datasets
adult_file_path = os.path.join(current_dir, 'Dataset', 'adult', 'adult', 'adult.data')
law_file_path = os.path.join(current_dir, 'Dataset', 'law', 'law.csv')
compas_file_path = os.path.join(current_dir, 'Dataset', 'compas', 'compas-scores-two-years.csv')
dutch_file_path = os.path.join(current_dir, 'Dataset', 'dutch', 'dutch_census_2001.csv')

def preprocess_dutch_census(file_path, protected_class='sex'):
    # Load the dataset (from CSV converted from ARFF)
    df = pd.read_csv(file_path)
    

    # Define the target variable
    target_column = 'occupation' 

    # Encode the target using LabelEncoder (e.g., 0 = non-prestigious, 1 = prestigious)
    y = LabelEncoder().fit_transform(df[target_column])

    # Drop the target column from the features
    features_df = df.drop(columns=[target_column])

    # Process the protected attribute
    if protected_class not in df.columns:
        raise ValueError(f"Protected class '{protected_class}' not found in dataset columns.")

    if protected_class == 'sex':
        # 1=male, 2=female → male=1 (privileged), female=0 (unprivileged)
        group = (df['sex'] == 1).astype(int).values
    elif protected_class == 'race':
        # White = privileged (1), others = 0
        group = (df['race'] == 'White').astype(int).values
    else:
        raise ValueError("Supported protected_class values are 'sex' or 'race'.")

    # Convert all object columns (categorical) to dummy variables
    categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()

    # Exclude the protected attribute from dummy encoding (if still in features)
    if protected_class in categorical_cols:
        categorical_cols.remove(protected_class)


    # One-hot encode
    features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=False)

    # Scale features using RobustScaler
    scaler = RobustScaler()
    x = scaler.fit_transform(features_df)

    return x, y, group



# def preprocess_adult_data(file_path, protected_class):
#     file_path = os.path.join(file_path, 'adult.csv')
#     columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
#                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
#     df = pd.read_csv(file_path, header=None, skipinitialspace=True, names=columns, skiprows=1)

#     # Handle missing values
#     df['workclass'] = df['workclass'].replace('?', np.nan)
#     df['occupation'] = df['occupation'].replace('?', np.nan)
#     df['native-country'] = df['native-country'].replace('?', np.nan)
#     df.dropna(inplace=True)

#     # Drop unnecessary columns
#     df.drop(['education-num', 'capital-gain', 'capital-loss', 'fnlwgt'], axis=1, inplace=True)

#     # Encode categorical variables
#     df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
#     df['race'] = df['race'].apply(lambda x: 1 if x == 'White' else 0)
#     df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
#     label_encoder = LabelEncoder()
#     categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
#     for column in categorical_columns:
#         df[column] = label_encoder.fit_transform(df[column])

#     # Apply RobustScaler to numerical columns
#     numerical_columns = ['age', 'hours-per-week']  # Add other numerical columns if needed
#     scaler = RobustScaler()
#     df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#     # Prepare output
#     y = df['income']
#     x = df.drop('income', axis=1)

#     if protected_class == 'sex':
#         group = df['sex']
#     elif protected_class == 'race':
#         group = df['race']

#     return x.to_numpy(), y.to_numpy(), group.to_numpy()

def preprocess_adult_data(data_dir, protected_class, use_test_data=False):
    # Define column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Load dataset
    file_path = data_dir
    df = pd.read_csv(file_path, header=None, names=columns, na_values=['?'], skipinitialspace=True)
    
    # Fix income labels for test set
    if use_test_data:
        df['income'] = df['income'].str.replace('.', '', regex=False)
    
    # Handle missing values using mode imputation
    for col in ['workclass', 'occupation', 'native-country']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode target variable
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})
    
    # Encode protected class
    if protected_class == 'sex':
        group = df['sex'].map({'Male': 1, 'Female': 0}).values
    elif protected_class == 'race':
        # Encode White as 1, non-White as 0
        group = (df['race'] == 'White').astype(int).values
    else:
        raise ValueError("Protected class must be 'sex' or 'race'")

    # Extract target variable BEFORE transforming DataFrame
    y = df['income'].values  # This line needs to be here!

    # Log-transform skewed features
    df['capital-gain'] = np.log1p(df['capital-gain'])
    df['capital-loss'] = np.log1p(df['capital-loss'])

    # Define numerical and categorical features
    numerical_cols = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
    categorical_cols = ['workclass', 'education', 'marital-status', 
                        'occupation', 'relationship', 'native-country']

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

    # Final feature matrix - create features array
    features = np.hstack((df[numerical_cols].values, df_encoded.values))

    # Standardize all features
    scaler = RobustScaler()
    x = scaler.fit_transform(features)

    return x, y, group




def preprocess_law_data(file_path, protected_class='race'):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Define target (pass_bar)
    y = df['pass_bar'].astype(int).values  # Target: pass_bar
    
    # Define features, including the protected class, but excluding 'pass_bar'
    features_df = df.drop(columns=['pass_bar'])  # Drop 'pass_bar' from features
    
    # Encode protected class
    if protected_class == 'sex':
        # Encode 'Male' as 1, 'Female' as 0
        group = df['male'].values
    elif protected_class == 'race':
        # Encode 'White' as 1, 'Non-White' as 0
        group = (df['race'] == 'White').astype(int).values
    else:
        raise ValueError("Protected class must be 'sex' or 'race'")
    
    # Convert 'race' column to numerical (1 for White, 0 for non-White)
    features_df['race'] = (features_df['race'] == 'White').astype(int)
    
    # Apply Robust Scaling to the features
    scaler = RobustScaler()
    x = scaler.fit_transform(features_df)
    
    return x, y, group




def preprocess_compas_data(data_dir, protected_class='race', random_state=1):

    # Define specific features to use
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"]
    CONT_VARIABLES = ["priors_count"]  # Continuous features
    CATEGORICAL_VARIABLES = ["age_cat", "c_charge_degree"]  # Categorical features to one-hot encode
    CLASS_FEATURE = "two_year_recid"  # Target variable
    
    # Ensure protected class is valid
    if protected_class not in ['race', 'sex']:
        raise ValueError("Protected class must be either 'race' or 'sex'")
    
    # Load the data
    file_path = data_dir
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find COMPAS dataset at {file_path}")
    
    # Load the dataset as a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Apply filters based on ProPublica's analysis
    # Filter 1: Arrest date within 30 days of COMPAS screening
    df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
    
    # Filter 2: Valid recidivism flag
    df = df[df['is_recid'] != -1]
    
    # Filter 3: Remove ordinary traffic offenses
    df = df[df['c_charge_degree'] != 'O']
    
    # Filter 4: Only include valid scores
    df = df[df['score_text'] != 'NA']
    
    # Filter 5: Only include African-American and Caucasian individuals if using race
    if protected_class == 'race':
        df = df[df['race'].isin(['African-American', 'Caucasian'])]
    
    # Extract the protected attribute FIRST (as requested)
    if protected_class == 'race':
        # For race: 1 for Caucasian (white), 0 for African-American (non-white)
        group = np.array([1 if val == 'Caucasian' else 0 for val in df['race']]).astype(int)
        print(f"Protected attribute (race): 1 = Caucasian (white), 0 = African-American (non-white)")
    elif protected_class == 'sex':
        # For sex: 1 for Male, 0 for Female
        group = np.array([1 if val == 'Male' else 0 for val in df['sex']]).astype(int)
        print(f"Protected attribute (sex): 1 = Male, 0 = Female")
    
    # Extract and process the target variable
    y = df[CLASS_FEATURE].values.astype(int)
    
    # Create feature matrix with one-hot encoding for categorical features
    features_df = pd.DataFrame()
    
    # Process continuous variables first (for clarity)
    for attr in CONT_VARIABLES:
        if attr in FEATURES_CLASSIFICATION:
            # Fill missing values with median
            df[attr] = df[attr].fillna(df[attr].median())
            features_df[attr] = df[attr].values
    
    # Process categorical variables with one-hot encoding
    for attr in CATEGORICAL_VARIABLES:
        if attr in FEATURES_CLASSIFICATION:
            # Create dummies (one-hot encoding)
            dummies = pd.get_dummies(df[attr], prefix=attr, drop_first=False)
            for col in dummies.columns:
                features_df[col] = dummies[col].values
    
    # Add protected attribute to features
    features_df[protected_class] = group
    
    # NOW apply robust scaling to ALL features, including categorical and protected
    scaler = RobustScaler()
    X = scaler.fit_transform(features_df.values)
    
    # Ensure X is 2D array and y and group are 1D arrays
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)  # Convert 1D to 2D if needed
    
    # Ensure y is 1D
    y = y.flatten()
    
    # Ensure group is 1D
    group = group.flatten()
    
    # Randomly permute the data (use same permutation for all)
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    group = group[perm]

    return X, y, group





def get_dataset(dataset='acsincome', protected_class='sex', batch_size=128, model=None):

    if 'acs' in dataset:
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=['CA'], download=True)

        # Mapping dataset names to Folktables task classes
        task_mapping = {
            'acsincome': ACSIncome,                 # 10 features
            'acsemployment': ACSEmployment,         # 17 features
            'acstraveltime': ACSTravelTime,         # 16 features
            'acspubliccoverage': ACSPublicCoverage, # 19 features
            'acsmobility': ACSMobility,             # 21 features
        }

        if dataset not in task_mapping:
            raise ValueError(f"Dataset {dataset} not recognized. Choose from: {list(task_mapping.keys())}")

        task_class = task_mapping[dataset]

        # Set protected attribute
        if protected_class == 'sex':
            task_class._group = 'SEX'
        elif protected_class == 'race':
            task_class._group = 'RAC1P'

        # Extract features, labels, and protected attribute
        features, label, group = task_class.df_to_numpy(acs_data)
        print(f"Dataset {dataset} loaded with {len(features)} samples, {features.shape[1]} features.")
        # Ensure protected attribute is binary
        if protected_class == 'sex':
            group[group == 2] = 0  # Convert Female (2) to 0
        elif protected_class == 'race':
            group[group > 1] = 0  # Group all non-white races together

        if dataset == 'acsincome':
            # Get a subset similar to adult/law dataset sizes
            subset_size = 50000  # Adjust this number as needed
            
            if len(features) > subset_size:
                # Simple random sampling
                np.random.seed(1)  # For reproducibility
                indices = np.random.choice(len(features), subset_size, replace=False)
                
                features = features[indices]
                label = label[indices]
                group = group[indices]
                
                print(f"Reduced acsincome dataset from {len(indices)} to {subset_size} samples")


    elif dataset == 'adult':
        features, label, group = preprocess_adult_data(adult_file_path, protected_class, use_test_data=False)
    elif dataset == 'law':
        features, label, group = preprocess_law_data(law_file_path, protected_class)
    elif dataset == 'compas':
        features, label, group = preprocess_compas_data(compas_file_path, protected_class)
    elif dataset == 'dutch':
        features, label, group = preprocess_dutch_census(dutch_file_path, protected_class)
    # Train-test split
    x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, test_size=0.2, random_state=args.init_seed)

    # Calculate and stores the mean and standard deviation of the data
    scaler = StandardScaler()
    scaler.fit(x_train)
    # Applies the transform to both training and test data
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train, x_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    group_train, group_test = torch.tensor(group_train, dtype=torch.float32), torch.tensor(group_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train, group_train)
    test_dataset = TensorDataset(x_test, y_test, group_test)

    if args.full_batch:
        # Use the full dataset as one batch
        batch_size = len(train_dataset)
        trainloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4 if args.device == "cuda" else 0,
            pin_memory=True if args.device == "cuda" else False
        )
        
        # TestLoader also as full batch
        testloader = DataLoader(
            dataset=test_dataset, 
            batch_size=len(test_dataset),
            shuffle=False,
            drop_last=True,
            num_workers=4 if args.device == "cuda" else 0,
            pin_memory=True if args.device == "cuda" else False
        )
    else:
        # Get indices for each group
        s0_indices = (group_train == 0).nonzero(as_tuple=True)[0].tolist()
        s1_indices = (group_train == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Number of samples in each group
        n_s0 = len(s0_indices)
        n_s1 = len(s1_indices)
        
        print(f"Demographics - S=0: {n_s0}, S=1: {n_s1}, Ratio: {n_s0/(n_s0+n_s1):.3f}:{n_s1/(n_s0+n_s1):.3f}")
        
        # Calculate what fraction of each group to include in each batch
        # based on batch size and total number of samples
        fraction = args.batch_size / (n_s0 + n_s1)
        
        # Ensure the fraction is reasonable (not too small or too large)
        fraction = min(0.5, max(0.01, fraction))
        
        # Calculate samples per batch from each group
        s0_per_batch = max(1, int(n_s0 * fraction))
        s1_per_batch = max(1, int(n_s1 * fraction))
        
        # Adjust to match batch size
        total = s0_per_batch + s1_per_batch
        if total != args.batch_size:
            # Scale to match batch size
            scale = args.batch_size / total
            s0_per_batch = max(1, int(s0_per_batch * scale))
            s1_per_batch = args.batch_size - s0_per_batch
        
        print(f"Each batch uses {s0_per_batch}/{n_s0} S=0 samples ({s0_per_batch/n_s0:.1%}) and {s1_per_batch}/{n_s1} S=1 samples ({s1_per_batch/n_s1:.1%})")
        print(f"Total batch size: {s0_per_batch + s1_per_batch}")
        
        # Calculate a reasonable number of batches per epoch
        # We want each epoch to use a significant portion of the dataset
        total_samples = n_s0 + n_s1
        samples_per_batch = args.batch_size
        num_batches = int(total_samples / samples_per_batch)
        

        def generate_balanced_batches():
            batches = []
            for _ in range(num_batches):
                batch = []
                
                # Sample from S=0 group with replacement
                batch.extend(random.choices(s0_indices, k=s0_per_batch))
                
                # Sample from S=1 group with replacement
                batch.extend(random.choices(s1_indices, k=s1_per_batch))
                
                # Shuffle to mix the groups
                random.shuffle(batch)
                batches.append(batch)
            
            return batches
        
        # Generate batches for training
        train_batches = generate_balanced_batches()
        
        trainloader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batches,
            num_workers=4 if args.device == "cuda" else 0,
            pin_memory=True if args.device == "cuda" else False
        )
        
        # Do the same for test data
        s0_indices_test = (group_test == 0).nonzero(as_tuple=True)[0].tolist()
        s1_indices_test = (group_test == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Function to generate test batches
        def generate_test_batches():
            batches = []
            for _ in range(10):  # Fewer batches for testing
                batch = []
                
                # Sample from S=0 group with replacement
                batch.extend(random.choices(s0_indices_test, k=s0_per_batch))
                
                # Sample from S=1 group with replacement
                batch.extend(random.choices(s1_indices_test, k=s1_per_batch))
                
                # Shuffle to mix the groups
                random.shuffle(batch)
                batches.append(batch)
            
            return batches
        
        # Generate batches for testing
        test_batches = generate_test_batches()
        
        testloader = DataLoader(
            dataset=test_dataset,
            batch_sampler=test_batches,
            num_workers=4 if args.device == "cuda" else 0,
            pin_memory=True if args.device == "cuda" else False
        )

    s_mean = group_train.mean().item()
    return trainloader, testloader, s_mean


# def get_dataset(dataset='acsemployment', protected_class='sex', batch_size=128, model=None):

#     if 'acs' in dataset:
#         data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
#         acs_data = data_source.get_data(states=['CA'], download=False)

#         # Mapping dataset names to Folktables task classes
#         task_mapping = {
#             'acsincome': ACSIncome,                 # 10 features, 1,664,500 datapoints
#             'acsemployment': ACSEmployment,         # 17 features, 3,236,107 datapoints
#             'acstraveltime': ACSTravelTime,         # 16 features, 1,466,648 datapoints
#             'acspubliccoverage': ACSPublicCoverage, # 19 features, 1,138,289 datapoints
#             'acsmobility': ACSMobility,             # 21 features, 620,937 datapoints
#         }

#         if dataset not in task_mapping:
#             raise ValueError(f"Dataset {dataset} not recognized. Choose from: {list(task_mapping.keys())}")

#         task_class = task_mapping[dataset]

#         # Set protected attribute
#         if protected_class == 'sex':
#             task_class._group = 'SEX'
#         elif protected_class == 'race':
#             task_class._group = 'RAC1P'

#         # Extract features, labels, and protected attribute
#         features, label, group = task_class.df_to_numpy(acs_data)

#         # Ensure protected attribute is binary
#         if protected_class == 'sex':
#             group[group == 2] = 0  # Convert Female (2) to 0
#         elif protected_class == 'race':
#             group[group > 1] = 0  # Group all non-white races together

#     elif dataset == 'adult':
#         features, label, group = preprocess_adult_data(adult_file_path, protected_class)
#     elif dataset == 'law':
#         features, label, group = preprocess_law_data(law_file_path, protected_class)
#     elif dataset == 'compas':
#         features, label, group = preprocess_compas_data(compas_file_path, protected_class)

#     # Train-test split
#     x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, test_size=0.2, random_state=1)

#     # Calculate and stores the mean and standard deviation of the data
#     scaler = StandardScaler()
#     scaler.fit(x_train)
#     # Applies the transform to both training and test data
#     x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

#     # Convert to PyTorch tensors
#     x_train, x_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
#     y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
#     group_train, group_test = torch.tensor(group_train, dtype=torch.float32), torch.tensor(group_test, dtype=torch.float32)

#     train_dataset = TensorDataset(x_train, y_train, group_train)
#     test_dataset = TensorDataset(x_test, y_test, group_test)

#     if args.full_batch:
#         batch_size = len(train_dataset)
#     else:
#         batch_size = args.batch_size
#     trainloader = DataLoader(
#         dataset=train_dataset, 
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=True,
#         num_workers=4 if args.device == "cuda" else 0,
#         pin_memory=True if args.device == "cuda" else False
#     )

#     if args.full_batch:
#         batch_size = len(test_dataset)
#     else:
#         batch_size = args.batch_size
#     # TestLoader remains as full batch
#     testloader = DataLoader(
#         dataset=test_dataset, 
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=True,
#         num_workers=4 if args.device == "cuda" else 0,
#         pin_memory=True if args.device == "cuda" else False
#     )

#     s_mean = group_train.mean().item()
#     # print(f"dataset size {len(train_dataset) + len(test_dataset)}")
#     return trainloader, testloader, s_mean


