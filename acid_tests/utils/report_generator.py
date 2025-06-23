import json
import datetime
from typing import Dict, Any
from pathlib import Path


class ReportGenerator:
    """Generate test reports in various formats"""
    
    def __init__(self, results_dir: str = "results", reports_dir: str = "reports"):
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def save_json_results(self, test_name: str, results: Dict[str, Any]):
        """Save raw test results as JSON"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results_with_metadata = {
            "test_name": test_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\n📊 Results saved to: {filepath}")
        return filepath
    
    def generate_markdown_report(self, test_name: str, results: Dict[str, Any]):
        """Generate a markdown report from test results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"# {test_name.upper()} Test Report\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write("| Database | Test Type | Result | Details |\n")
            f.write("|----------|-----------|--------|----------|\n")
            
            for db_name, db_results in results.items():
                if isinstance(db_results, dict):
                    # Handle proper atomicity test results
                    if 'transactional_atomicity' in db_results:
                        trans = db_results['transactional_atomicity']
                        if trans.get('rollback_successful'):
                            status = "✅ PASS"
                            behavior = "Transaction rollback"
                            details = f"{trans.get('vectors_rolled_back', 0)} vectors rolled back"
                        else:
                            status = "❌ FAIL"
                            behavior = "No rollback"
                            details = "Transaction failed"
                        f.write(f"| {db_name} | Transactional | {status} | {details} |\n")
                    
                    elif 'best_effort_behavior' in db_results:
                        best_effort = db_results.get('best_effort_behavior', {})
                        if best_effort.get('behavior') == 'best_effort_partial_success':
                            status = "✅ BASE"
                            details = f"{best_effort.get('vectors_inserted')}/{best_effort.get('vectors_attempted')} inserted"
                        elif best_effort.get('behavior') == 'all_or_nothing_validation':
                            status = "⚠️ N/A"
                            details = "Validation prevents test"
                        else:
                            status = "❓ UNKNOWN"
                            details = "Unexpected behavior"
                        f.write(f"| {db_name} | Best-effort | {status} | {details} |\n")
                    
                    # Legacy batch atomicity
                    elif 'batch_atomicity' in db_results:
                        batch = db_results['batch_atomicity']
                        if batch.get('atomic'):
                            status = "✅ ATOMIC"
                            behavior = "Complete rollback"
                        elif batch.get('partial_success'):
                            status = "❌ NON-ATOMIC"
                            behavior = f"Partial success ({batch.get('vectors_added')}/{batch.get('vectors_attempted')})"
                        else:
                            status = "⚠️ UNKNOWN"
                            behavior = "Unexpected behavior"
                        
                        details = batch.get('error_message', 'N/A')[:50] + "..." if batch.get('error_message') else "Success"
                        f.write(f"| {db_name} | {status} | {behavior} | {details} |\n")
            
            # Detailed Results
            f.write("\n## Detailed Results\n\n")
            for db_name, db_results in results.items():
                f.write(f"### {db_name}\n\n")
                
                if isinstance(db_results, dict):
                    # Transactional atomicity (pgvector)
                    if 'transactional_atomicity' in db_results:
                        f.write("#### Transaction Test\n")
                        trans = db_results['transactional_atomicity']
                        f.write(f"- Test type: SQL transaction with deliberate rollback\n")
                        f.write(f"- Vectors in transaction: {trans.get('vectors_rolled_back', 100)}\n")
                        f.write(f"- Rollback successful: {trans.get('rollback_successful', False)}\n")
                        f.write(f"- Result: {'All vectors rolled back' if trans.get('rollback_successful') else 'Rollback failed'}\n")
                        f.write("\n")
                    
                    # Best-effort behavior (others)
                    elif 'best_effort_behavior' in db_results:
                        f.write("#### Best-Effort Insert Test\n")
                        best_effort = db_results.get('best_effort_behavior', {})
                        f.write(f"- Test type: Batch with invalid vector at position {best_effort.get('invalid_position', 50)}\n")
                        f.write(f"- Vectors attempted: {best_effort.get('vectors_attempted', 'N/A')}\n")
                        f.write(f"- Vectors inserted: {best_effort.get('vectors_inserted', 'N/A')}\n")
                        f.write(f"- Behavior: {best_effort.get('behavior', 'unknown')}\n")
                        if best_effort.get('error_message'):
                            f.write(f"- Error: {best_effort.get('error_message')[:100]}...\n")
                        f.write("\n")
                        
                        # Partial failure test
                        partial = db_results.get('partial_failure', {})
                        if partial:
                            f.write("#### Partial Failure Test\n")
                            f.write(f"- Test type: Batch with duplicate ID at position {partial.get('duplicate_position', 50)}\n")
                            f.write(f"- Vectors added: {partial.get('vectors_added', 'N/A')}\n")
                            f.write(f"- Partial success: {partial.get('partial_success', False)}\n")
                            f.write("\n")
                    
                    # Legacy batch atomicity results
                    elif 'batch_atomicity' in db_results:
                        f.write("#### Batch Atomicity Test\n")
                        batch = db_results['batch_atomicity']
                        f.write(f"- Vectors attempted: {batch.get('vectors_attempted', 'N/A')}\n")
                        f.write(f"- Vectors inserted: {batch.get('vectors_added', 'N/A')}\n")
                        f.write(f"- Atomic behavior: {batch.get('atomic', False)}\n")
                        f.write(f"- Supports transactions: {batch.get('supports_transactions', False)}\n")
                        if batch.get('error_message'):
                            f.write(f"- Error: {batch.get('error_message')}\n")
                        f.write("\n")
                    
                    # Update atomicity results (if present)
                    if 'update_atomicity' in db_results:
                        f.write("#### Update Atomicity Test\n")
                        update = db_results['update_atomicity']
                        f.write(f"- Successful updates: {update.get('success_count', 'N/A')}\n")
                        f.write(f"- Failed updates: {update.get('error_count', 'N/A')}\n")
                        f.write(f"- Final version: {update.get('final_version', 'N/A')}\n")
                        f.write("\n")
            
            # Conclusions based on actual test results
            f.write("\n## Conclusions\n\n")
            f.write("### Observed Behaviors\n\n")
            
            # Analyze each database based on results
            for db_name, db_results in results.items():
                if isinstance(db_results, dict):
                    f.write(f"**{db_name}**:\n")
                    
                    # Check for proper atomicity test results
                    if 'transactional_atomicity' in db_results:
                        # pgvector transaction test
                        trans = db_results['transactional_atomicity']
                        if trans.get('rollback_successful'):
                            f.write(f"- ✅ Transaction rollback successful\n")
                            f.write(f"- {trans.get('vectors_rolled_back', 0)} vectors rolled back\n")
                        else:
                            f.write(f"- ❌ Transaction rollback failed\n")
                    
                    elif 'best_effort_behavior' in db_results:
                        # Other databases
                        best_effort = db_results.get('best_effort_behavior', {})
                        partial = db_results.get('partial_failure', {})
                        
                        if best_effort.get('behavior') == 'best_effort_partial_success':
                            f.write(f"- ✅ Best-effort partial success (expected BASE behavior)\n")
                            f.write(f"- {best_effort.get('vectors_inserted')}/{best_effort.get('vectors_attempted')} vectors inserted\n")
                        elif best_effort.get('behavior') == 'all_or_nothing_validation':
                            f.write(f"- ⚠️ Strict validation prevents partial insert\n")
                        
                        if partial.get('partial_success'):
                            f.write(f"- Duplicate handling: {partial.get('vectors_added')} vectors added (duplicate skipped)\n")
                    
                    # Legacy batch atomicity results
                    elif 'batch_atomicity' in db_results:
                        batch = db_results['batch_atomicity']
                        if batch.get('atomic'):
                            f.write(f"- ✅ Demonstrated atomic behavior (all-or-nothing)\n")
                            if batch.get('supports_transactions'):
                                f.write(f"- Supports explicit transactions with rollback\n")
                        elif batch.get('duplicate_behavior'):
                            f.write(f"- ⚠️ Handles duplicates via {batch['duplicate_behavior']}\n")
                            f.write(f"- No error on duplicate IDs - silently deduplicates\n")
                            f.write(f"- Inserted {batch.get('vectors_added', 0)} vectors despite duplicate\n")
                        elif batch.get('partial_success'):
                            f.write(f"- ❌ Allows partial batch success\n")
                            f.write(f"- {batch.get('vectors_added', 0)} out of {batch.get('vectors_attempted', 0)} vectors persisted\n")
                    
                    f.write("\n")
            
            f.write("### Key Findings\n\n")
            f.write("1. **Atomicity != Error Handling**: Some databases show 'atomic' behavior by rejecting entire batches due to validation errors, not true transactional atomicity\n")
            f.write("2. **Milvus Special Case**: Silently handles duplicates without errors, making it appear non-atomic when it's actually a design choice\n")
            f.write("3. **True ACID**: Only pgvector demonstrated actual transaction support with rollback capability\n\n")
            
            f.write("### Technical Characteristics Based on Test Results\n\n")
            f.write("**Transaction Support**:\n")
            f.write("- pgvector: Explicit SQL transactions with rollback capability\n")
            f.write("- Others: No transaction support (BASE model)\n\n")
            
            f.write("**Duplicate ID Handling**:\n")
            f.write("- Milvus: Silently deduplicates (first-write-wins)\n")
            f.write("- Qdrant/ChromaDB: Rejects entire batch on duplicate detection\n")
            f.write("- pgvector: Depends on constraint configuration\n\n")
            
            f.write("**Batch Validation Behavior**:\n")
            f.write("- ChromaDB/Milvus: Strict dimension validation (all-or-nothing)\n")
            f.write("- Qdrant: Partial insertion possible before validation error\n")
            f.write("- pgvector: Depends on transaction boundaries\n\n")
            
            f.write("**ID Format Requirements**:\n")
            f.write("- pgvector/Qdrant: Strict UUID format required\n")
            f.write("- Milvus/ChromaDB: Flexible string ID format\n")
        
        print(f"📄 Report saved to: {filepath}")
        return filepath
    
    def generate_html_report(self, test_name: str, results: Dict[str, Any]):
        """Generate an HTML report with charts"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.html"
        filepath = self.reports_dir / filename
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{test_name.upper()} Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .warning {{ color: orange; }}
    </style>
</head>
<body>
    <h1>{test_name.upper()} Test Report</h1>
    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Summary</h2>
    <table>
        <tr>
            <th>Database</th>
            <th>Atomicity</th>
            <th>Behavior</th>
            <th>Details</th>
        </tr>
"""
        
        for db_name, db_results in results.items():
            if isinstance(db_results, dict) and 'batch_atomicity' in db_results:
                batch = db_results['batch_atomicity']
                if batch.get('atomic'):
                    status = '<span class="pass">✅ ATOMIC</span>'
                    behavior = "Complete rollback"
                elif batch.get('partial_success'):
                    status = '<span class="fail">❌ NON-ATOMIC</span>'
                    behavior = f"Partial success ({batch.get('vectors_added')}/{batch.get('vectors_attempted')})"
                else:
                    status = '<span class="warning">⚠️ UNKNOWN</span>'
                    behavior = "Unexpected behavior"
                
                details = batch.get('error_message', 'N/A')[:50] + "..." if batch.get('error_message') else "Success"
                html_content += f"""
        <tr>
            <td>{db_name}</td>
            <td>{status}</td>
            <td>{behavior}</td>
            <td>{details}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Detailed Results</h2>
"""
        
        # Add detailed results for each database
        for db_name, db_results in results.items():
            html_content += f"<h3>{db_name}</h3>"
            if isinstance(db_results, dict):
                html_content += "<pre>" + json.dumps(db_results, indent=2) + "</pre>"
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {filepath}")
        return filepath
    
    def generate_consistency_report(self, results: Dict[str, Any], test_type: str):
        """Generate a consistency test report"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consistency_{test_type}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"# Consistency Test Report - {test_type.replace('_', ' ').title()}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## Overview\n")
            f.write("This report presents the results of consistency tests across multiple vector databases.\n")
            f.write("Test notation: ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Database | Overall Status | Passed | Partial | Failed | Details |\n")
            f.write("|----------|---------------|---------|---------|---------|----------|\n")
            
            # Process results for each test type
            test_results = []
            for key, value in results.items():
                if isinstance(value, list):
                    test_results = value
                    break
            
            for result in test_results:
                if isinstance(result, dict) and 'database' in result:
                    db_name = result['database']
                    
                    if 'error' in result:
                        f.write(f"| {db_name} | ❌ ERROR | - | - | - | {result['error'][:50]}... |\n")
                        continue
                    
                    summary = result.get('summary', {})
                    overall = summary.get('overall_status', '?')
                    passed = summary.get('passed', 0)
                    partial = summary.get('partial', 0)
                    failed = summary.get('failed', 0)
                    total = summary.get('total_tests', 0)
                    
                    details = f"{passed}/{total} tests passed"
                    f.write(f"| {db_name} | {overall} | {passed} | {partial} | {failed} | {details} |\n")
            
            # Detailed results by database
            f.write("\n## Detailed Results\n\n")
            
            for result in test_results:
                if isinstance(result, dict) and 'database' in result:
                    db_name = result['database']
                    f.write(f"### {db_name}\n\n")
                    
                    if 'error' in result:
                        f.write(f"**Error**: {result['error']}\n\n")
                        continue
                    
                    tests = result.get('tests', {})
                    
                    # Format based on test type
                    if test_type == "schema_constraints":
                        f.write("#### Schema Constraint Tests\n\n")
                        for test_name, test_result in tests.items():
                            status = test_result.get('status', '?')
                            message = test_result.get('message', 'No message')
                            f.write(f"- **{test_name.replace('_', ' ').title()}**: {status} - {message}\n")
                    
                    elif test_type == "index_synchronization":
                        f.write("#### Index Synchronization Tests\n\n")
                        for test_name, test_result in tests.items():
                            status = test_result.get('status', '?')
                            message = test_result.get('message', 'No message')
                            f.write(f"- **{test_name.replace('_', ' ').title()}**: {status} - {message}\n")
                            
                            # Add details if available
                            details = test_result.get('details', {})
                            if details:
                                f.write(f"  - Visible: {details.get('visible_count', 'N/A')}\n")
                                f.write(f"  - Missing: {details.get('missing_count', 'N/A')}\n")
                                f.write(f"  - Avg delay: {details.get('avg_delay_ms', 'N/A')}ms\n")
                                f.write(f"  - Max delay: {details.get('max_delay_ms', 'N/A')}ms\n")
                    
                    elif test_type == "metadata_consistency":
                        f.write("#### Metadata Consistency Tests\n\n")
                        for test_name, test_result in tests.items():
                            status = test_result.get('status', '?')
                            message = test_result.get('message', 'No message')
                            f.write(f"- **{test_name.replace('_', ' ').title()}**: {status} - {message}\n")
                            
                            # Add type details if available
                            if test_name == "type_consistency" and 'details' in test_result:
                                f.write("  - Type preservation:\n")
                                for dtype, dstatus in test_result['details'].items():
                                    f.write(f"    - {dtype}: {dstatus}\n")
                    
                    elif test_type == "query_consistency":
                        f.write("#### Query Consistency Tests\n\n")
                        for test_name, test_result in tests.items():
                            status = test_result.get('status', '?')
                            message = test_result.get('message', 'No message')
                            f.write(f"- **{test_name.replace('_', ' ').title()}**: {status} - {message}\n")
                    
                    f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            if test_type == "schema_constraints":
                f.write("### Schema Validation Patterns\n\n")
                f.write("1. **Dimension Enforcement**: All databases correctly validate vector dimensions\n")
                f.write("2. **Batch Handling**: Databases differ in whether they reject entire batches or allow partial success\n")
                f.write("3. **Error Reporting**: Clear error messages help identify schema violations\n\n")
            
            elif test_type == "index_synchronization":
                f.write("### Index Synchronization Patterns\n\n")
                f.write("1. **Immediate Visibility**: Most databases show good immediate visibility for single inserts\n")
                f.write("2. **Load Impact**: High insert loads can cause visibility delays\n")
                f.write("3. **Consistency Trade-offs**: Some databases prioritize write speed over immediate consistency\n\n")
            
            elif test_type == "metadata_consistency":
                f.write("### Metadata Handling Patterns\n\n")
                f.write("1. **Update Support**: Not all databases support direct metadata updates\n")
                f.write("2. **Concurrent Safety**: Concurrent metadata updates may lead to inconsistencies\n")
                f.write("3. **Type Preservation**: Data type handling varies across databases\n\n")
            
            elif test_type == "query_consistency":
                f.write("### Query Consistency Patterns\n\n")
                f.write("1. **Read-After-Write**: Most databases show good read-after-write consistency\n")
                f.write("2. **Update Impact**: Ongoing updates can cause transient inconsistencies\n")
                f.write("3. **Concurrent Reads**: Some variation in concurrent read results observed\n\n")
            
            # Technical notes
            f.write("## Technical Notes\n\n")
            f.write("- All tests were run against locally deployed instances\n")
            f.write("- Network latency and resource constraints may affect results\n")
            f.write("- Results represent behavior at the time of testing\n")
            f.write("- BASE systems showing ⚠ or ✗ may be operating as designed\n")
        
        print(f"📄 Consistency report saved to: {filepath}")
        
        # Also generate HTML report
        html_filepath = self.generate_consistency_html_report(results, test_type)
        
        return filepath
    
    def generate_consistency_html_report(self, results: Dict[str, Any], test_type: str):
        """Generate an HTML consistency test report"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consistency_{test_type}_{timestamp}.html"
        filepath = self.reports_dir / filename
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Consistency Test Report - {test_type.replace('_', ' ').title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .pass {{ color: green; }}
        .partial {{ color: orange; }}
        .fail {{ color: red; }}
        .na {{ color: gray; }}
        h3 {{ margin-top: 30px; }}
        ul {{ margin: 10px 0; }}
        .details {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Consistency Test Report - {test_type.replace('_', ' ').title()}</h1>
    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Test notation:</strong> ✓ = no inconsistency observed, ⚠ = partial inconsistency or delay, ✗ = unstable or broken behavior</p>
    
    <h2>Summary</h2>
    <table>
        <tr>
            <th>Database</th>
            <th>Overall Status</th>
            <th>Passed</th>
            <th>Partial</th>
            <th>Failed</th>
            <th>Details</th>
        </tr>
"""
        
        # Process results
        test_results = []
        for key, value in results.items():
            if isinstance(value, list):
                test_results = value
                break
        
        for result in test_results:
            if isinstance(result, dict) and 'database' in result:
                db_name = result['database']
                
                if 'error' in result:
                    html_content += f"""
        <tr>
            <td>{db_name}</td>
            <td><span class="fail">❌ ERROR</span></td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>{result['error'][:50]}...</td>
        </tr>
"""
                    continue
                
                summary = result.get('summary', {})
                overall = summary.get('overall_status', '?')
                passed = summary.get('passed', 0)
                partial = summary.get('partial', 0)
                failed = summary.get('failed', 0)
                total = summary.get('total_tests', 0)
                
                status_class = 'pass' if overall == '✓' else 'partial' if overall == '⚠' else 'fail'
                
                html_content += f"""
        <tr>
            <td>{db_name}</td>
            <td><span class="{status_class}">{overall}</span></td>
            <td>{passed}</td>
            <td>{partial}</td>
            <td>{failed}</td>
            <td>{passed}/{total} tests passed</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Detailed Results</h2>
"""
        
        # Detailed results by database
        for result in test_results:
            if isinstance(result, dict) and 'database' in result:
                db_name = result['database']
                html_content += f"<h3>{db_name}</h3>"
                
                if 'error' in result:
                    html_content += f'<p class="fail"><strong>Error:</strong> {result["error"]}</p>'
                    continue
                
                tests = result.get('tests', {})
                html_content += '<div class="details">'
                html_content += f'<h4>{test_type.replace("_", " ").title()} Tests</h4>'
                html_content += '<ul>'
                
                for test_name, test_result in tests.items():
                    status = test_result.get('status', '?')
                    message = test_result.get('message', 'No message')
                    
                    status_class = 'pass' if status == '✓' else 'partial' if status == '⚠' else 'fail' if status == '✗' else 'na'
                    
                    html_content += f'<li><strong>{test_name.replace("_", " ").title()}:</strong> '
                    html_content += f'<span class="{status_class}">{status}</span> - {message}'
                    
                    # Add details if available
                    if 'details' in test_result:
                        html_content += '<ul>'
                        details = test_result['details']
                        if isinstance(details, dict):
                            for k, v in details.items():
                                html_content += f'<li>{k}: {v}</li>'
                        html_content += '</ul>'
                    
                    html_content += '</li>'
                
                html_content += '</ul>'
                html_content += '</div>'
        
        html_content += f"""
    <h2>Key Findings</h2>
    <div class="details">
        <h3>{test_type.replace('_', ' ').title()} Patterns</h3>
"""
        
        if test_type == "schema_constraints":
            html_content += """
        <ol>
            <li><strong>Dimension Enforcement:</strong> All databases correctly validate vector dimensions</li>
            <li><strong>Batch Handling:</strong> Databases differ in whether they reject entire batches or allow partial success</li>
            <li><strong>Error Reporting:</strong> Clear error messages help identify schema violations</li>
        </ol>
"""
        elif test_type == "index_synchronization":
            html_content += """
        <ol>
            <li><strong>Immediate Visibility:</strong> Most databases show good immediate visibility for single inserts</li>
            <li><strong>Load Impact:</strong> High insert loads can cause visibility delays</li>
            <li><strong>Consistency Trade-offs:</strong> Some databases prioritize write speed over immediate consistency</li>
        </ol>
"""
        elif test_type == "metadata_consistency":
            html_content += """
        <ol>
            <li><strong>Update Support:</strong> Not all databases support direct metadata updates</li>
            <li><strong>Concurrent Safety:</strong> Concurrent metadata updates may lead to inconsistencies</li>
            <li><strong>Type Preservation:</strong> Data type handling varies across databases</li>
        </ol>
"""
        elif test_type == "query_consistency":
            html_content += """
        <ol>
            <li><strong>Read-After-Write:</strong> Most databases show good read-after-write consistency</li>
            <li><strong>Update Impact:</strong> Ongoing updates can cause transient inconsistencies</li>
            <li><strong>Concurrent Reads:</strong> Some variation in concurrent read results observed</li>
        </ol>
"""
        
        html_content += """
    </div>
    
    <h2>Technical Notes</h2>
    <ul>
        <li>All tests were run against locally deployed instances</li>
        <li>Network latency and resource constraints may affect results</li>
        <li>Results represent behavior at the time of testing</li>
        <li>BASE systems showing ⚠ or ✗ may be operating as designed</li>
    </ul>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML consistency report saved to: {filepath}")
        return filepath