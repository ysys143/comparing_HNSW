#!/usr/bin/env python3
"""
Generate HTML reports from existing JSON results
"""
import json
import sys
from pathlib import Path
from utils.report_generator import ReportGenerator

def main():
    # Find all consistency JSON result files
    results_dir = Path("results")
    report_gen = ReportGenerator()
    
    consistency_files = list(results_dir.glob("consistency_*.json"))
    
    print(f"Found {len(consistency_files)} consistency test results")
    
    for json_file in consistency_files:
        print(f"\nProcessing: {json_file}")
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Determine test type from filename
        if "schema" in json_file.name:
            test_type = "schema_constraints"
        elif "index" in json_file.name:
            test_type = "index_synchronization"
        elif "metadata" in json_file.name:
            test_type = "metadata_consistency"
        elif "query" in json_file.name:
            test_type = "query_consistency"
        else:
            print(f"Unknown test type for {json_file}")
            continue
        
        # Generate HTML report
        try:
            results = {test_type: data['results']}
            html_path = report_gen.generate_consistency_html_report(results, test_type)
            print(f"Generated: {html_path}")
        except Exception as e:
            print(f"Error generating HTML for {json_file}: {str(e)}")

if __name__ == "__main__":
    main()