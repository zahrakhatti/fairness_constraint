import os
import re
import glob
import pandas as pd

def extract_eps_lambda_from_filename(filename):
    """Extract epsilon and lambda values from filename."""
    # Remove .csv extension
    filename = filename.replace('.csv', '')
    
    # Extract values using regex
    eps_pattern = r'eps_([^_]+)'
    lambda_pattern = r'lambda_([^_]+)'
    
    eps_match = re.search(eps_pattern, filename)
    lambda_match = re.search(lambda_pattern, filename)
    
    if not eps_match or not lambda_match:
        return None, None
    
    eps_str = eps_match.group(1)
    lambda_str = lambda_match.group(1)
    
    # Convert values
    epsilon = float('inf') if eps_str.lower() == 'inf' else float(eps_str)
    lmbd = float('inf') if lambda_str.lower() == 'inf' else float(lambda_str)
            
    return epsilon, lmbd

def find_csv_files(directory):
    """Find all CSV files in a directory and its subdirectories."""
    if not os.path.exists(directory):
        return []
    
    return [os.path.join(root, file) 
            for root, _, files in os.walk(directory) 
            for file in files if file.endswith('.csv')]

def find_base_directory(args):
    """Find the base directory for results."""
    model_type = args.model_types[0] if hasattr(args, 'model_types') and args.model_types else 2
    
    if args.scaled == False:
        # Try both directory patterns
        possible_paths = [
            f"output/scaled_{args.scaled}/{args.dataset}/full_batch_{args.full_batch}/{model_type}/{args.data_plot}_results"
        ]
    else:
        possible_paths = [
            f"output/{args.dataset}/full_batch_{args.full_batch}/{model_type}/{args.data_plot}_results"
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found valid directory: {path}")
            return path
    
    if args.scaled == False:
        # Check parent directories
        parent_paths = [
            f"output/scaled_{args.scaled}/{args.dataset}/full_batch_{args.full_batch}/{model_type}/{args.data_plot}_results"
        ]
    else:
        parent_paths = [
            f"output/{args.dataset}/full_batch_{args.full_batch}/{model_type}/{args.data_plot}_results"
        ]
    
    for path in parent_paths:
        if os.path.exists(path):
            print(f"Found results directory: {path}")
            return path 
    
    # Default to first path
    print(f"No directory found, using default: {possible_paths[0]}")
    return possible_paths[0]

def get_result_files(args):
    """Get result files and base directory."""
    base_dir = find_base_directory(args)
    os.makedirs(base_dir, exist_ok=True)
    
    # Get all CSV files
    result_files = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    
    # Try recursive search if no files found
    if not result_files:
        print(f"No files found directly in {base_dir}, searching recursively...")
        result_files = find_csv_files(base_dir)
    
    print(f"Found {len(result_files)} result files in {base_dir}")
    return result_files, base_dir

def read_csv_with_comments(file_path):
    """Read CSV file while preserving comment lines."""
    comments = []
    data_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                comments.append(line.rstrip())
            else:
                data_lines.append(line)
    
    # Extract header line (first non-comment line)
    header_line = None
    data_content = []
    
    for i, line in enumerate(data_lines):
        if i == 0:
            header_line = line.strip()
        else:
            data_content.append(line)
    
    # Now parse the data using pandas
    import io
    data_str = header_line + '\n' + ''.join(data_content)
    df = pd.read_csv(io.StringIO(data_str), comment='#', engine='python')
    
    return comments, df

def normalize_spaces(text):
    """Remove extra spaces between column values in aligned CSV format."""
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text.strip())

def parse_aligned_csv(file_path):
    """Parse CSV files with aligned columns and spacing."""
    comments = []
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                comments.append(line.rstrip())
            else:
                # Normalize spaces and split by comma
                normalized = normalize_spaces(line)
                data.append(normalized)
    
    # Extract header and data rows
    if not data:
        return comments, pd.DataFrame()
    
    header = data[0]
    rows = data[1:]
    
    # Convert to DataFrame
    df = pd.DataFrame([row.split(',') for row in rows], columns=header.split(','))
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Convert numeric columns to appropriate types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    return comments, df

def parse_results(args):
    """
    Parse all result files and return a combined dataframe.
    
    Args:
        args: Configuration arguments
        
    Returns:
        pd.DataFrame: Combined dataframe with all results
    """
    result_files, _ = get_result_files(args)
    
    if not result_files:
        print("No result files found.")
        return pd.DataFrame()
    
    data_frames = []
    all_comments = {}
    
    for file_path in result_files:
        try:
            filename = os.path.basename(file_path)
            epsilon, lmbd = extract_eps_lambda_from_filename(filename)
            
            if epsilon is None or lmbd is None:
                print(f"Skipping file, could not extract parameters: {filename}")
                continue
            
            # Read CSV file with comments
            comments, df = parse_aligned_csv(file_path)
            
            # Store comments for this file
            all_comments[filename] = comments
            
            # Add epsilon and lambda columns
            df['epsilon'] = epsilon
            df['lmbd'] = lmbd
            
            data_frames.append(df)
            print(f"Processed file {filename}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not data_frames:
        return pd.DataFrame(), all_comments
    
    # Combine all dataframes
    combined_df = pd.concat(data_frames, ignore_index=True)
    print(f"Combined dataframe shape: {combined_df.shape}")
    return combined_df, all_comments

def write_aligned_csv(df, output_path, comments=None):
    """
    Write a DataFrame to a CSV file with aligned columns.
    
    Args:
        df: DataFrame to write
        output_path: Path to write the file to
        comments: List of comment lines to include at the top
    """
    # Write comments
    with open(output_path, 'w') as f:
        if comments:
            for comment in comments:
                f.write(f"{comment}\n")
        
        # Calculate column widths based on values and headers
        column_widths = {}
        for col in df.columns:
            # Get max width of column values
            values = df[col].astype(str)
            max_value_width = values.str.len().max()
            
            # Compare with header width
            column_widths[col] = max(max_value_width, len(col)) + 2
        
        # Write header
        header_line = ", ".join([col.ljust(column_widths[col]) for col in df.columns])
        f.write(header_line + "\n")
        
        # Write data rows
        for _, row in df.iterrows():
            row_values = []
            for col in df.columns:
                value = str(row[col])
                row_values.append(value.ljust(column_widths[col]))
            
            line = ", ".join(row_values)
            f.write(line + "\n")

def read_aligned_and_convert(input_path, output_path=None):
    """
    Read an aligned CSV file and optionally convert it to a properly formatted version.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to write the converted file to (optional)
        
    Returns:
        pd.DataFrame: The parsed DataFrame
    """
    comments, df = parse_aligned_csv(input_path)
    
    if output_path:
        # Write the DataFrame to a proper CSV file
        df.to_csv(output_path, index=False)
        print(f"Converted file written to {output_path}")
        
        # Also write an aligned version if requested
        aligned_output = output_path.replace('.csv', '_aligned.csv')
        write_aligned_csv(df, aligned_output, comments)
        print(f"Aligned version written to {aligned_output}")
    
    return df, comments

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse and process aligned CSV files')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    if args.input:
        df, comments = read_aligned_and_convert(args.input, args.output)
        print(f"Parsed DataFrame with {len(df)} rows and {len(df.columns)} columns")
    else:
        print("Please provide an input file path")