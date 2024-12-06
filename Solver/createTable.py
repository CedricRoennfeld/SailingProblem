from Classes.SailingLeagueProblem import SailingLeagueProblem

# WARNING: File may not work after changes in SailingLeagueProblem.py (Parameters)

# Function to print T[k][r] as a transposed r-k table
def print_table(T):
    # Get the range of k and r dynamically from the dictionary keys
    k_values = sorted(T.keys())  # Extract and sort all k values
    r_values = sorted({r for k in T for r in T[k].keys()})  # Extract and sort all r values

    # Print the header row (k values)
    print(f"{'r/k':<5}", end="")
    for k in k_values:
        print(f"{k:<5}", end="")
    print()  # Newline after header

    # Print each row for each r
    for r in r_values:
        print(f"{r:<5}", end="")  # Row label for r
        for k in k_values:
            # Check if T[k] contains the key r, otherwise print a placeholder (e.g., '-')
            if r in T[k]:
                if T[k][r][1] == 0:
                    print(f"{T[k][r][0]:<5}", end="")  # Value T[k][r]
                else:
                    print(f"~{T[k][r][0]:<5}", end="")
            else:
                print(f"{'-':<5}", end="")  # Placeholder for missing values
        print()  # Newline after each row

def create_table(value_ranges: dict, timelimit):
    T = dict()
    for k,(r_lb, r_ub) in value_ranges.items():
        T[k] = dict()
        for r in range(r_lb,r_ub+1):
            print(f"analyzing k={k}, r={r}")
            SLP = SailingLeagueProblem(2*k,r)
            SLP.optimize(timelimit=timelimit, output_flag=False)
            SLP.save_results(f"Results/result_{k}_{r}.rslt")
            T[k][r] = [int(SLP.opj_val),SLP.obj_gap]
            print_table(T)

    return T