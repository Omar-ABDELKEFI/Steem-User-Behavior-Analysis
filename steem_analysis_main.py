import pandas as pd
import glob
import os

def merge_and_deduplicate_csv(directory, pattern="steem_accounts_analysis_*.csv", chunk_size=10000):
    """Merges all CSV files matching the pattern with memory-efficient approach,
       removes duplicates based on 'Username', and prints the DataFrame length."""

    all_files = glob.glob(os.path.join(directory, pattern))

    if not all_files:
        print(f"No files found matching the pattern '{pattern}' in the directory.")
        return None

    print(f"Found {len(all_files)} files to process.")

    # First, collect all unique usernames to filter out duplicates early
    usernames_seen = set()
    total_rows = 0
    final_df = None

    # Process each file individually
    for i, file in enumerate(all_files):
        print(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(file)}")
        try:
            # Read file in chunks to avoid memory issues
            chunk_list = []
            for chunk in pd.read_csv(file, chunksize=chunk_size):
                total_rows += len(chunk)
                # Only keep rows with usernames we haven't seen before
                chunk = chunk[~chunk['Username'].isin(usernames_seen)]
                if not chunk.empty:
                    chunk_list.append(chunk)
                    # Update our set of seen usernames
                    usernames_seen.update(chunk['Username'].tolist())

            # Combine chunks from this file
            if chunk_list:
                file_df = pd.concat(chunk_list, ignore_index=True)
                # Append to final dataframe
                if final_df is None:
                    final_df = file_df
                else:
                    final_df = pd.concat([final_df, file_df], ignore_index=True)
                print(f"  - Added {len(file_df)} unique rows from this file")
                print(f"  - Current unique usernames: {len(usernames_seen)}")
            else:
                print("  - No new unique usernames in this file")

        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file {file}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if final_df is None:
        print("No valid data found to merge.")
        return None

    print(f"\nSummary:")
    print(f"Total rows processed across all files: {total_rows}")
    print(f"Final DataFrame length (unique usernames): {len(final_df)}")

    return final_df

def save_and_display_sample(df, output_path, sample_rows=20):
    """Saves the dataframe to a file and returns a sample for display"""
    if df is not None:
        # Save the merged data
        df.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")

        # Return a sample for display
        return df.head(sample_rows)
    return None

# Example usage
directory_path = '/content/drive/MyDrive/final-resultat-steem'
output_file = os.path.join(directory_path, 'merged_steem_accounts.csv')

# Run the merge operation
merged_df = merge_and_deduplicate_csv(directory_path)

# Save complete results and display a sample
if merged_df is not None:
    sample_df = save_and_display_sample(merged_df, output_file)
    print("\nSample of merged data:")
    display(sample_df)
