#!/usr/bin/env python3
"""
Master Verification Script
Runs all verification scripts and generates comprehensive audit report.

This script:
1. Runs all verification scripts that generate JSON results
2. Validates that all tex claims are backed by results
3. Generates a comprehensive verification summary

Generated files:
- verification_summary.json
- All individual *_results.json files
"""

import json
import os
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

# Directories
CODE_DIR = Path(__file__).parent
RESULTS_DIR = CODE_DIR.parent / 'results'

# Verification scripts to run (in order)
VERIFICATION_SCRIPTS = [
    'sign_test_multi_event.py',
    'economic_magnitude_calculations.py',
    'market_structure_outage.py',
    'wallet_markout_persistence.py',
    'randomization_inference_outage.py',
    'rolling_oos_toxicity.py',
]

def run_script(script_name):
    """Run a verification script and capture results."""
    script_path = CODE_DIR / script_name

    if not script_path.exists():
        return {
            'script': script_name,
            'status': 'NOT_FOUND',
            'error': f'Script not found at {script_path}'
        }

    try:
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            script_name.replace('.py', ''),
            script_path
        )
        module = importlib.util.module_from_spec(spec)

        # Execute the main function
        spec.loader.exec_module(module)

        if hasattr(module, 'main'):
            result = module.main()
            return {
                'script': script_name,
                'status': 'SUCCESS',
                'result_keys': list(result.keys()) if isinstance(result, dict) else None
            }
        else:
            return {
                'script': script_name,
                'status': 'NO_MAIN',
                'error': 'No main() function found'
            }

    except Exception as e:
        return {
            'script': script_name,
            'status': 'ERROR',
            'error': str(e)
        }

def collect_verification_status():
    """Collect verification status from all JSON results files."""

    verification_claims = []

    # Files to check for tex_verification sections
    results_files = [
        'sign_test_results.json',
        'economic_magnitude_results.json',
        'market_structure_outage_results.json',
        'wallet_markout_persistence_results.json',
        'randomization_inference_results.json',
        'rolling_oos_toxicity_results.json',
        'clustered_inference_main_results.json',
        'outage_event_study_results.json',
        'jan2025_congestion_results.json',
        'oos_classification_results.json',
        'mpsc_lob_validated_results.json',
        'tob_fragility_verification.json',
        'wallet_identity_robustness.json',
        'iv_within_event_results.json',
    ]

    for filename in results_files:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract tex_verification if present
                if 'tex_verification' in data:
                    tv = data['tex_verification']
                    if 'claims_verified' in tv:
                        for claim in tv['claims_verified']:
                            verification_claims.append({
                                'source_file': filename,
                                'claim': claim.get('claim', 'Unknown'),
                                'value': claim.get('value', claim.get('computed', 'N/A')),
                                'verified': claim.get('verified', False)
                            })
            except (json.JSONDecodeError, KeyError) as e:
                pass

    return verification_claims

def generate_summary():
    """Generate comprehensive verification summary."""

    print("=" * 70)
    print("MASTER VERIFICATION SCRIPT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    # Run all verification scripts
    print("Running verification scripts...")
    print("-" * 70)

    script_results = []
    for script in VERIFICATION_SCRIPTS:
        print(f"  Running {script}...", end=" ")
        result = run_script(script)
        script_results.append(result)
        print(f"[{result['status']}]")

    print()

    # Collect all verification claims
    print("Collecting verification claims...")
    print("-" * 70)

    claims = collect_verification_status()

    n_verified = sum(1 for c in claims if c['verified'])
    n_total = len(claims)

    print(f"  Total claims found: {n_total}")
    print(f"  Verified: {n_verified}")
    print(f"  Unverified: {n_total - n_verified}")
    print()

    # Summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'scripts_run': len(VERIFICATION_SCRIPTS),
        'scripts_success': sum(1 for r in script_results if r['status'] == 'SUCCESS'),
        'scripts_failed': sum(1 for r in script_results if r['status'] != 'SUCCESS'),
        'total_claims': n_total,
        'verified_claims': n_verified,
        'unverified_claims': n_total - n_verified,
        'verification_rate': round(n_verified / n_total * 100, 1) if n_total > 0 else 0,
        'script_results': script_results,
        'claims_by_file': {}
    }

    # Group claims by source file
    for claim in claims:
        src = claim['source_file']
        if src not in summary['claims_by_file']:
            summary['claims_by_file'][src] = {
                'total': 0,
                'verified': 0,
                'claims': []
            }
        summary['claims_by_file'][src]['total'] += 1
        if claim['verified']:
            summary['claims_by_file'][src]['verified'] += 1
        summary['claims_by_file'][src]['claims'].append(claim)

    # Print summary by file
    print("Verification by file:")
    print("-" * 70)
    for filename, data in summary['claims_by_file'].items():
        status = "✓" if data['verified'] == data['total'] else "⚠"
        print(f"  {status} {filename}: {data['verified']}/{data['total']} verified")

    print()
    print("=" * 70)
    print(f"OVERALL VERIFICATION RATE: {summary['verification_rate']}%")
    print("=" * 70)

    return summary

def main():
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # Generate summary
    summary = generate_summary()

    # Save summary
    output_path = RESULTS_DIR / 'verification_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nVerification summary saved to: {output_path}")

    # Return summary for programmatic access
    return summary

if __name__ == "__main__":
    main()
