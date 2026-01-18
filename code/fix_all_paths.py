#!/usr/bin/env python3
"""
Fix all hardcoded absolute paths to use relative paths.
Makes the project 100% portable and reproducible.
"""
import os
import re

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Files with Path('/Users/...') patterns
files_to_fix = [
    'iv_correct_analysis.py',
    'concentration_fragility_regression.py',
    'lag_structure_rigorous.py',
    'cross_event_learning_iv.py',
    'out_of_sample_classification.py',
    'combined_iv_analysis.py',
    'outage_robust_inference.py',
    'download_wallet_data.py',
    'horizon_persistence_analysis.py',
    'infrastructure_identity_analysis.py',
    'wallet_identity_robustness.py',
    'api_outage_comprehensive.py',
    'lag_structure_endogeneity_analysis.py',
    'jfe_strengthening_analysis.py',
    'api_outage_analysis.py',
    'mpsc_lob_validated.py',
    'run_full_analysis.py',
    'jan2025_congestion_analysis.py',
    'rd_validation_comprehensive.py',
    'iv_informed_share_analysis.py',
    'tob_fragility_verification.py',
    'food_chain_analysis.py',
    'lag_structure_refined.py',
    'multi_event_infrastructure_shocks.py',
    'popcat_exploration.py',
    'informed_classification_validation.py',
    'tick_rd_comprehensive.py',
    'mpsc_analysis.py',
    'api_outage_event_study.py',
    'counterparty_analysis_robust.py',
    'tob_fragility_alternative.py',
    'multi_event_fragility_analysis.py',
    'wallet_concentration_analysis.py',
    'mpsc_withdrawal_fragility.py',
    'iv_within_event_analysis.py',
    'popcat_micro_analysis.py',
]

def fix_file(filepath):
    """Fix hardcoded paths in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Standard header to add relative path setup
    rel_path_setup = '''
# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
'''
    
    # Replace various hardcoded OUTPUT_DIR patterns
    patterns = [
        # Path('/Users/advaitjayant/...')
        (r"OUTPUT_DIR\s*=\s*Path\(['\"]\/Users\/advaitjayant[^'\"]+['\"]\)", 
         "OUTPUT_DIR = Path(_RESULTS_DIR)"),
        (r"DATA_DIR\s*=\s*Path\(['\"]\/Users\/advaitjayant[^'\"]+['\"]\)",
         "DATA_DIR = Path(_DATA_DIR)"),
        (r"FIGURES_DIR\s*=\s*Path\(['\"]\/Users\/advaitjayant[^'\"]+['\"]\)",
         "FIGURES_DIR = Path(_FIGURES_DIR)"),
        (r"RESULTS_DIR\s*=\s*Path\(['\"]\/Users\/advaitjayant[^'\"]+['\"]\)",
         "RESULTS_DIR = Path(_RESULTS_DIR)"),
        (r"CODE_DIR\s*=\s*Path\(['\"]\/Users\/advaitjayant[^'\"]+['\"]\)",
         "CODE_DIR = Path(_CODE_DIR)"),
        
        # String '/Users/advaitjayant/...'
        (r"OUTPUT_DIR\s*=\s*['\"]\/Users\/advaitjayant[^'\"]+['\"]",
         "OUTPUT_DIR = _RESULTS_DIR"),
        (r"DATA_DIR\s*=\s*['\"]\/Users\/advaitjayant[^'\"]+['\"]",
         "DATA_DIR = _DATA_DIR"),
        
        # Inline hardcoded paths
        (r"['\"]\/Users\/advaitjayant\/Downloads\/Boyi PhD\/Second year paper\/The Liquidity-Efficiency Paradox\/figures\/([^'\"]+)['\"]",
         r"os.path.join(_FIGURES_DIR, '\1')"),
        (r"['\"]\/Users\/advaitjayant\/Downloads\/Boyi PhD\/Second year paper\/The Liquidity-Efficiency Paradox\/wallet_data\/([^'\"]+)['\"]",
         r"os.path.join(_DATA_DIR, '\1')"),
        (r"['\"]\/Users\/advaitjayant\/Downloads\/Boyi PhD\/Second year paper\/The Liquidity-Efficiency Paradox\/([^'\"\/]+\.parquet)['\"]",
         r"os.path.join(_DATA_DIR, '\1')"),
        (r"['\"]\/Users\/advaitjayant\/Downloads\/Boyi PhD\/Second year paper\/The Liquidity-Efficiency Paradox\/([^'\"\/]+\.csv)['\"]",
         r"os.path.join(_RESULTS_DIR, '\1')"),
    ]
    
    # Check if already has relative path setup
    if '_PROJECT_ROOT' not in content and '/Users/advaitjayant' in content:
        # Find import block and add after it
        import_match = re.search(r'^((?:import [^\n]+\n|from [^\n]+\n)+)', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.end()
            content = content[:insert_pos] + rel_path_setup + content[insert_pos:]
    
    # Apply all replacements
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    fixed_count = 0
    for filename in files_to_fix:
        filepath = os.path.join(CODE_DIR, filename)
        if os.path.exists(filepath):
            if fix_file(filepath):
                print(f"Fixed: {filename}")
                fixed_count += 1
            else:
                print(f"No changes: {filename}")
        else:
            print(f"Not found: {filename}")
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()
