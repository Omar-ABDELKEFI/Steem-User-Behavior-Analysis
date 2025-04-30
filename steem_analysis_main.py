import os
import csv
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback
import sys
import time

# Configuration
STEEMWORLD_API_BASE = 'https://sds.steemworld.org/transfers_api/getTransfers'
FETCH_LIMIT = 1000

# Default values
DEFAULT_USERNAMES_FILE = "/content/drive/MyDrive/steem-suer-analysis/sorted_usernames_cleaned.csv"
DEFAULT_START_DATE = "2025-03-01"
DEFAULT_END_DATE = "2025-03-31"

def read_usernames_from_csv(csv_path):
    """Read usernames from a CSV file"""
    try:
        # Try using pandas to read the file, which is more robust for different CSV formats
        df = pd.read_csv(csv_path)

        # Check if there's a column named 'username'
        username_column = None
        for col in df.columns:
            if col.lower() == 'username':
                username_column = col
                break

        if username_column:
            # Extract usernames from the username column
            usernames = df[username_column].dropna().astype(str).str.strip().tolist()
        else:
            # If no username column, assume the first column contains usernames
            usernames = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        # Remove empty strings
        usernames = [u for u in usernames if u]
        return usernames

    except Exception as e:
        # Fall back to the original CSV reader if pandas fails
        usernames = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Check if there's a column named 'username'
            username_index = -1
            if header:
                for i, col in enumerate(header):
                    if col.lower() == 'username':
                        username_index = i
                        break

            if username_index >= 0:
                # There's a username column
                for row in reader:
                    if len(row) > username_index and row[username_index].strip():
                        usernames.append(row[username_index].strip())
            else:
                # Assume each line is a username
                if header and len(header) == 1:  # If the first row is likely a username
                    usernames.append(header[0])
                for row in reader:
                    if row and row[0].strip():
                        usernames.append(row[0].strip())

        return usernames

def find_username_index(usernames, target_username):
    """
    Find the index of a username in the list, with case-insensitive search
    """
    # Try exact match first
    try:
        return usernames.index(target_username)
    except ValueError:
        # Try case-insensitive match
        lowercase_usernames = [u.lower() for u in usernames]
        try:
            lowercase_index = lowercase_usernames.index(target_username.lower())
            return lowercase_index
        except ValueError:
            return -1

def format_date(date_str):
    """Format date string to YYYY-MM-DD format"""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

def date_to_timestamp(date_str):
    """Convert date string to Unix timestamp"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

def is_within_last_week(timestamp):
    """Check if timestamp is within the last 7 days"""
    tx_date = datetime.fromtimestamp(timestamp)
    cutoff_date = datetime.now() - timedelta(days=7)
    return tx_date >= cutoff_date

def calculate_total_amount(operations):
    """Calculate total amount from operations"""
    total = 0
    for op in operations:
        if isinstance(op['amount'], str):
            parts = op['amount'].split(' ')
            amount = float(parts[0]) if parts else 0
        elif isinstance(op['amount'], (int, float)):
            amount = float(op['amount'])
        else:
            amount = 0
        total += amount
    return total

def fetch_account_operations(account, date_range, max_retries=3):
    """Fetch account operations from SteemWorld API with retries"""
    from_time = date_to_timestamp(date_range['from'])
    to_time = date_to_timestamp(date_range['to'])

    # Prepare API requests for both operation types
    transfer_params = {
        "type": "transfer",
        "orderBy": "time",
        "orderDir": "DESC",
        "from": account,
        "fromTime": from_time,
        "toTime": to_time
    }

    power_up_params = {
        "type": "transfer_to_vesting",
        "orderBy": "time",
        "orderDir": "DESC",
        "from": account,
        "fromTime": from_time,
        "toTime": to_time
    }

    # Encode parameters for API URL
    transfer_params_encoded = json.dumps(transfer_params)
    power_up_params_encoded = json.dumps(power_up_params)

    # Build API URLs
    transfer_url = f"{STEEMWORLD_API_BASE}/{transfer_params_encoded}/{FETCH_LIMIT}/0"
    power_up_url = f"{STEEMWORLD_API_BASE}/{power_up_params_encoded}/{FETCH_LIMIT}/0"

    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            # Fetch data
            transfer_response = requests.get(transfer_url)
            transfer_data = transfer_response.json()

            power_up_response = requests.get(power_up_url)
            power_up_data = power_up_response.json()

            # Process transfer operations
            transfers = []
            if transfer_data.get('code') == 0 and transfer_data.get('result', {}).get('rows'):
                cols = transfer_data['result']['cols']
                transfers = [
                    {
                        'timestamp': row[cols['time']],
                        'from': row[cols['from']],
                        'to': row[cols['to']],
                        'amount': row[cols['amount']],
                        'currency': row[cols['unit']],
                        'memo': row[cols['memo']] if 'memo' in cols else '',
                        'type': 'transfer'
                    }
                    for row in transfer_data['result']['rows']
                ]

            # Process power-up operations
            power_ups = []
            if power_up_data.get('code') == 0 and power_up_data.get('result', {}).get('rows'):
                cols = power_up_data['result']['cols']
                power_ups = [
                    {
                        'timestamp': row[cols['time']],
                        'from': row[cols['from']],
                        'to': row[cols['to']],
                        'amount': row[cols['amount']],
                        'currency': row[cols['unit']],
                        'memo': '',
                        'type': 'power_up'
                    }
                    for row in power_up_data['result']['rows']
                ]

            # Filter for withdrawals
            withdrawals = [tx for tx in transfers if tx['from'] == account and tx['to'] != account]

            # Calculate statistics
            total_powered_up = calculate_total_amount(power_ups)
            total_withdrawn = calculate_total_amount(withdrawals)

            # Filter operations from the last week
            last_week_power_ups = [op for op in power_ups if is_within_last_week(op['timestamp'])]
            last_week_withdrawals = [tx for tx in withdrawals if is_within_last_week(tx['timestamp'])]

            last_week_power_up_total = calculate_total_amount(last_week_power_ups)
            last_week_withdrawal_total = calculate_total_amount(last_week_withdrawals)

            return {
                'powerUps': power_ups,
                'withdrawals': withdrawals,
                'stats': {
                    'totalPoweredUp': total_powered_up,
                    'totalWithdrawn': total_withdrawn,
                    'lastWeekPowerUpTotal': last_week_power_up_total,
                    'lastWeekWithdrawalTotal': last_week_withdrawal_total
                }
            }
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            time.sleep(1)  # Wait for 1 second before retrying

    # If we reach here, all retries have failed
    return {
        'powerUps': [],
        'withdrawals': [],
        'stats': {
            'totalPoweredUp': 0,
            'totalWithdrawn': 0,
            'lastWeekPowerUpTotal': 0,
            'lastWeekWithdrawalTotal': 0
        },
        'error': last_error
    }

def analyze_account(username, date_range, max_retries=10):
    """Analyze a single Steem account using real API interaction"""
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            operations = fetch_account_operations(username, date_range)

            if 'error' in operations:
                last_error = operations['error']
                retry_count += 1
                time.sleep(1)  # Wait for 1 second before retrying
                continue

            # Extract amounts as numbers
            power_up_total = operations['stats']['totalPoweredUp']
            withdrawal_total = operations['stats']['totalWithdrawn']

            # Calculate total operations
            total_amount = power_up_total + withdrawal_total

            # Calculate percentages
            power_up_percentage = 0
            withdrawal_percentage = 0

            if total_amount > 0:
                power_up_percentage = (power_up_total / total_amount) * 100
                withdrawal_percentage = (withdrawal_total / total_amount) * 100

            # Determine behavior category
            category = "No Activity"

            if total_amount > 0:
                if power_up_percentage == 100:
                    category = "100% Power Up"
                elif power_up_percentage >= 75:
                    category = "75% Power Up"
                elif power_up_percentage >= 50:
                    category = "50% Power Up"
                elif power_up_percentage >= 25:
                    category = "25% Power Up"
                else:
                    category = "Primarily Withdrawal"

            return {
                "username": username,
                "powerUpTotal": power_up_total,
                "withdrawalTotal": withdrawal_total,
                "totalAmount": total_amount,
                "powerUpPercentage": power_up_percentage,
                "withdrawalPercentage": withdrawal_percentage,
                "category": category,
                "success": True,
                "error": None
            }

        except Exception as e:
            last_error = str(e)
            retry_count += 1
            time.sleep(1)  # Wait for 1 second before retrying

    # If we reach here, all retries have failed
    return {
        "username": username,
        "powerUpTotal": 0,
        "withdrawalTotal": 0,
        "totalAmount": 0,
        "powerUpPercentage": 0,
        "withdrawalPercentage": 0,
        "category": "No Activity",
        "success": False,
        "error": last_error
    }

def analyze_all_accounts(usernames, date_range, num_threads=10):
    """Analyze multiple accounts in parallel using threading with tqdm progress bar"""
    results = {
        'accounts': [],
        'categories': {
            "100% Power Up": 0,
            "75% Power Up": 0,
            "50% Power Up": 0,
            "25% Power Up": 0,
            "Primarily Withdrawal": 0,
            "No Activity": 0,
            "Error": 0  # Added for tracking errors
        },
        'totalPowerUp': 0,
        'totalWithdrawal': 0,
        'totalAccounts': 0
    }

    # Thread-safe counter and progress tracking
    processed = {'count': 0}
    total = len(usernames)

    # Lock for thread-safe updates to the results dictionary
    lock = threading.Lock()

    # Create a fixed-position progress bar that doesn't create multiple lines
    pbar = tqdm(total=total, desc=f"Analyzing {total} accounts", unit="acc", leave=True)

    def process_account(username):
        try:
            # Analyze the account
            account_result = analyze_account(username, date_range)

            # Update the results dictionary in a thread-safe manner
            with lock:
                results['accounts'].append(account_result)

                if account_result.get('success', False):
                    category = account_result.get('category', 'No Activity')
                    results['categories'][category] += 1
                    results['totalPowerUp'] += account_result.get('powerUpTotal', 0)
                    results['totalWithdrawal'] += account_result.get('withdrawalTotal', 0)
                else:
                    # Increment the error counter
                    results['categories']["Error"] += 1

                # Update the progress counter
                processed['count'] += 1
                pbar.update(1)

        except Exception as e:
            with lock:
                results['categories']["Error"] += 1
                processed['count'] += 1
                pbar.update(1)

    try:
        # Process accounts using a thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(process_account, username) for username in usernames]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                # Just wait for completion, the progress bar is updated in process_account
                pass
    except Exception as e:
        pass
    finally:
        # Close the progress bar
        pbar.close()

    # Set the total accounts analyzed
    results['totalAccounts'] = len(results['accounts'])

    return results

def generate_csv_from_accounts(accounts):
    """Generate CSV content from account data"""
    csv_content = []

    # Header row - Restructured as per requirements
    header = ["Account Name", "Power Up Total", "Withdrawal Total", "Total Amount",
              "Power Up Percentage", "Withdrawal Percentage", "Behavior Category", "Error"]
    csv_content.append(header)

    # Data rows
    for account in accounts:
        username = account.get("username", "")
        power_up_total = account.get("powerUpTotal", 0)
        withdrawal_total = account.get("withdrawalTotal", 0)
        total_amount = account.get("totalAmount", 0)
        power_up_percentage = account.get("powerUpPercentage", 0)
        withdrawal_percentage = account.get("withdrawalPercentage", 0)
        category = account.get("category", "No Activity")
        error = account.get("error", "") if not account.get("success", True) else ""

        row = [username, power_up_total, withdrawal_total, total_amount,
               round(power_up_percentage, 1), round(withdrawal_percentage, 1),
               category, error]
        csv_content.append(row)

    return csv_content

def process_accounts_in_batches(usernames, date_range, start_index=0, batch_size=10000, output_dir="/content/drive/MyDrive/", num_threads=10):
    """Process accounts in batches and save results periodically using threading"""
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        output_dir = os.getcwd()

    # Initialize overall results for final summary only
    all_results = {
        'accounts': [],
        'categories': {
            "100% Power Up": 0,
            "75% Power Up": 0,
            "50% Power Up": 0,
            "25% Power Up": 0,
            "Primarily Withdrawal": 0,
            "No Activity": 0,
            "Error": 0
        },
        'totalPowerUp': 0,
        'totalWithdrawal': 0,
        'totalAccounts': 0
    }

    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_number = 1

    total_accounts = len(usernames)
    remaining_accounts = total_accounts - start_index

    # Process accounts in batches
    for batch_start in range(start_index, total_accounts, batch_size):
        batch_end = min(batch_start + batch_size, total_accounts)
        current_batch = usernames[batch_start:batch_end]

        # Analyze the current batch using threading
        batch_results = analyze_all_accounts(current_batch, date_range, num_threads)

        # Save only this batch's results to CSV
        batch_accounts_csv = generate_csv_from_accounts(batch_results['accounts'])
        batch_accounts_csv_path = os.path.join(output_dir, f"steem_accounts_analysis_{timestamp_base}_batch{batch_number}.csv")

        try:
            with open(batch_accounts_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(batch_accounts_csv)
        except Exception:
            pass

        # Merge batch results with overall results for final summary
        all_results['accounts'].extend(batch_results['accounts'])
        for category, count in batch_results['categories'].items():
            all_results['categories'][category] += count
        all_results['totalPowerUp'] += batch_results['totalPowerUp']
        all_results['totalWithdrawal'] += batch_results['totalWithdrawal']
        all_results['totalAccounts'] += batch_results['totalAccounts']

        batch_number += 1

    last_analyzed_account = usernames[batch_end-1] if 'batch_end' in locals() else ""

    return all_results, last_analyzed_account

def get_date_range():
    """Interactive function to get date range from user with helpful options"""
    print("\nDate Range Selection:")
    print("1. Last 7 days")
    print("2. Last 30 days")
    print("3. Last 90 days")
    print("4. Last 180 days")
    print("5. Last 365 days")
    print("6. Custom date range")
    print("7. Default date range (2025-03-01 to 2025-03-31)")

    choice = input("\nSelect an option (1-7) or press Enter for default (option 7): ").strip()

    today = datetime.now().strftime("%Y-%m-%d")

    if not choice or choice == '7':  # Default is the specified date range
        return {'from': DEFAULT_START_DATE, 'to': DEFAULT_END_DATE}

    if choice == '1':
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        return {'from': from_date, 'to': today}
    elif choice == '2':
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        return {'from': from_date, 'to': today}
    elif choice == '3':
        from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        return {'from': from_date, 'to': today}
    elif choice == '4':
        from_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        return {'from': from_date, 'to': today}
    elif choice == '5':
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        return {'from': from_date, 'to': today}
    elif choice == '6':
        print("\nEnter custom date range (format: YYYY-MM-DD):")
        while True:
            from_date = input("Start date: ").strip()
            try:
                format_date(from_date)
                break
            except ValueError as e:
                print(f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}")

        while True:
            to_date = input("End date (press Enter for today): ").strip()
            if not to_date:
                to_date = today
                break
            try:
                format_date(to_date)

                if datetime.strptime(to_date, "%Y-%m-%d") < datetime.strptime(from_date, "%Y-%m-%d"):
                    print("End date cannot be before start date. Please try again.")
                    continue
                break
            except ValueError as e:
                print(f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}")

        return {'from': from_date, 'to': to_date}
    else:
        print("Invalid option, using default (2025-03-01 to 2025-03-31).")
        return {'from': DEFAULT_START_DATE, 'to': DEFAULT_END_DATE}

def main():
    try:
        default_output_dir = "/content/drive/MyDrive/final-resultat-steem"
        output_dir = input(f"Enter output directory path (press Enter for default: {default_output_dir}): ").strip()
        if not output_dir:
            output_dir = default_output_dir

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            output_dir = os.getcwd()

        csv_path = input(f"Enter path to CSV file containing usernames (press Enter for default: {DEFAULT_USERNAMES_FILE}): ").strip()
        if not csv_path:
            csv_path = DEFAULT_USERNAMES_FILE

        if not os.path.isfile(csv_path):
            print(f"Error: File {csv_path} not found.")
            return

        date_range = get_date_range()
        print(f"Using date range: {date_range['from']} to {date_range['to']}")

        try:
            usernames = read_usernames_from_csv(csv_path)
        except Exception as e:
            print(f"Error reading usernames from CSV: {str(e)}")
            return

        if not usernames:
            print("Error: No usernames found in the CSV file.")
            return

        print(f"Found {len(usernames)} usernames.")

        max_accounts = input(f"How many accounts to analyze (1-{len(usernames)}, or press Enter for all): ")
        if max_accounts:
            try:
                max_accounts = int(max_accounts)
                usernames = usernames[:max_accounts]
            except ValueError:
                pass

        start_from_username = input("Start from a specific username? (Enter username or press Enter to start from beginning): ").strip()
        start_index = 0

        if start_from_username:
            username_index = find_username_index(usernames, start_from_username)
            if username_index >= 0:
                start_index = username_index
                print(f"Starting from {usernames[start_index]} (index {start_index}).")
            else:
                print("Starting from the beginning.")

        batch_size = 10000  # Default batch size
        custom_batch_size = input(f"Enter batch size for saving results (press Enter for default {batch_size}): ").strip()
        if custom_batch_size:
            try:
                batch_size = int(custom_batch_size)
                if batch_size <= 0:
                    batch_size = 10000
            except ValueError:
                pass

        num_threads = 5  # Default number of threads
        custom_threads = input(f"Enter number of threads for parallel processing (press Enter for default {num_threads}): ").strip()
        if custom_threads:
            try:
                num_threads = int(custom_threads)
                if num_threads <= 0:
                    num_threads = 5
            except ValueError:
                pass

        print(f"Analyzing accounts using {num_threads} threads...")
        try:
            analysis_results, last_analyzed_account = process_accounts_in_batches(usernames, date_range, start_index, batch_size, output_dir, num_threads)
        except Exception as e:
            print(f"Error processing accounts: {str(e)}")
            return

=        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accounts_csv = generate_csv_from_accounts(analysis_results['accounts'])

        final_accounts_csv_path = os.path.join(output_dir, f"steem_accounts_analysis_final_{timestamp}.csv")

        try:
            with open(final_accounts_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(accounts_csv)

            print(f"Analysis complete. Final results saved to {final_accounts_csv_path}")
            print("\nSummary:")
            for category, count in analysis_results['categories'].items():
                if analysis_results['totalAccounts'] > 0:
                    percentage = round((count / analysis_results['totalAccounts']) * 100, 1)
                else:
                    percentage = 0
                print(f"- {category}: {count} accounts ({percentage}%)")
        except Exception as e:
            print(f"Error saving final results: {str(e)}")

    except Exception as e:
        print(f"Unexpected error in main function: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)