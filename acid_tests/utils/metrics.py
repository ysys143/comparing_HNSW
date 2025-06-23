import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from pathlib import Path


@dataclass
class TestResult:
    """Individual test result"""
    database: str
    test_name: str
    test_category: str  # atomicity, consistency, isolation, durability
    score: float
    violations: int = 0
    duration_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ACIDMetrics:
    """ACID compliance metrics for a database"""
    database: str
    atomicity_score: float = 0.0
    consistency_score: float = 0.0
    isolation_score: float = 0.0
    durability_score: float = 0.0
    overall_score: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult):
        """Add a test result and update scores"""
        self.test_results.append(result)
        self._update_scores()
    
    def _update_scores(self):
        """Update category scores based on test results"""
        categories = {
            'atomicity': [],
            'consistency': [],
            'isolation': [],
            'durability': []
        }
        
        for result in self.test_results:
            if result.test_category in categories:
                categories[result.test_category].append(result.score)
        
        # Calculate average scores
        if categories['atomicity']:
            self.atomicity_score = np.mean(categories['atomicity'])
        if categories['consistency']:
            self.consistency_score = np.mean(categories['consistency'])
        if categories['isolation']:
            self.isolation_score = np.mean(categories['isolation'])
        if categories['durability']:
            self.durability_score = np.mean(categories['durability'])
        
        # Calculate overall score (weighted average)
        scores = []
        weights = []
        
        if self.atomicity_score > 0:
            scores.append(self.atomicity_score)
            weights.append(1.0)
        if self.consistency_score > 0:
            scores.append(self.consistency_score)
            weights.append(1.0)
        if self.isolation_score > 0:
            scores.append(self.isolation_score)
            weights.append(1.0)
        if self.durability_score > 0:
            scores.append(self.durability_score)
            weights.append(1.0)
        
        if scores:
            self.overall_score = np.average(scores, weights=weights)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics"""
        return {
            'database': self.database,
            'scores': {
                'atomicity': round(self.atomicity_score, 2),
                'consistency': round(self.consistency_score, 2),
                'isolation': round(self.isolation_score, 2),
                'durability': round(self.durability_score, 2),
                'overall': round(self.overall_score, 2)
            },
            'test_count': len(self.test_results),
            'total_violations': sum(r.violations for r in self.test_results),
            'failed_tests': sum(1 for r in self.test_results if r.score < 50),
            'timestamp': datetime.now().isoformat()
        }


class MetricsCollector:
    """Collect and analyze ACID test metrics"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.databases: Dict[str, ACIDMetrics] = {}
        self.start_time = time.time()
    
    def add_test_result(self, result: TestResult):
        """Add a test result"""
        if result.database not in self.databases:
            self.databases[result.database] = ACIDMetrics(database=result.database)
        
        self.databases[result.database].add_result(result)
    
    def get_database_metrics(self, database: str) -> Optional[ACIDMetrics]:
        """Get metrics for a specific database"""
        return self.databases.get(database)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive ACID report"""
        report = {
            'test_suite': 'ACID Compliance Tests',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - self.start_time,
            'databases': {}
        }
        
        # Add database summaries
        for db_name, metrics in self.databases.items():
            report['databases'][db_name] = metrics.get_summary()
        
        # Add comparative analysis
        report['comparison'] = self._generate_comparison()
        
        # Add test details
        report['test_details'] = self._get_test_details()
        
        return report
    
    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate comparative analysis"""
        comparison = {
            'rankings': {},
            'best_in_category': {},
            'compliance_matrix': {}
        }
        
        # Calculate rankings for each category
        for category in ['atomicity', 'consistency', 'isolation', 'durability', 'overall']:
            scores = []
            for db_name, metrics in self.databases.items():
                score = getattr(metrics, f"{category}_score")
                scores.append((db_name, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            comparison['rankings'][category] = scores
            
            if scores:
                comparison['best_in_category'][category] = scores[0][0]
        
        # Create compliance matrix
        for db_name, metrics in self.databases.items():
            comparison['compliance_matrix'][db_name] = {
                'atomicity': self._get_compliance_level(metrics.atomicity_score),
                'consistency': self._get_compliance_level(metrics.consistency_score),
                'isolation': self._get_compliance_level(metrics.isolation_score),
                'durability': self._get_compliance_level(metrics.durability_score)
            }
        
        return comparison
    
    def _get_compliance_level(self, score: float) -> str:
        """Convert score to compliance level"""
        if score >= 95:
            return "FULL"
        elif score >= 80:
            return "HIGH"
        elif score >= 60:
            return "PARTIAL"
        elif score >= 40:
            return "LOW"
        else:
            return "NONE"
    
    def _get_test_details(self) -> List[Dict[str, Any]]:
        """Get detailed test results"""
        all_tests = []
        
        for db_name, metrics in self.databases.items():
            for result in metrics.test_results:
                test_dict = asdict(result)
                test_dict['timestamp'] = datetime.fromtimestamp(
                    result.timestamp
                ).isoformat()
                all_tests.append(test_dict)
        
        # Sort by timestamp
        all_tests.sort(key=lambda x: x['timestamp'])
        
        return all_tests
    
    def save_report(self, filename: str = "acid_report.json"):
        """Save report to file"""
        report = self.generate_report()
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_path}")
        
        # Also save a summary CSV
        self._save_summary_csv()
    
    def _save_summary_csv(self):
        """Save summary as CSV for easy viewing"""
        csv_path = self.output_dir / "acid_summary.csv"
        
        with open(csv_path, 'w') as f:
            # Header
            f.write("Database,Atomicity,Consistency,Isolation,Durability,Overall\n")
            
            # Data
            for db_name, metrics in self.databases.items():
                f.write(f"{db_name},"
                       f"{metrics.atomicity_score:.1f},"
                       f"{metrics.consistency_score:.1f},"
                       f"{metrics.isolation_score:.1f},"
                       f"{metrics.durability_score:.1f},"
                       f"{metrics.overall_score:.1f}\n")
        
        print(f"Summary CSV saved to: {csv_path}")
    
    def print_summary(self):
        """Print summary to console"""
        print("\n" + "="*60)
        print("ACID COMPLIANCE TEST SUMMARY")
        print("="*60)
        
        # Print table header
        print(f"\n{'Database':<15} {'Atomicity':>10} {'Consistency':>12} "
              f"{'Isolation':>10} {'Durability':>11} {'Overall':>8}")
        print("-"*80)
        
        # Print scores for each database
        for db_name, metrics in self.databases.items():
            print(f"{db_name:<15} "
                  f"{metrics.atomicity_score:>9.1f}% "
                  f"{metrics.consistency_score:>11.1f}% "
                  f"{metrics.isolation_score:>9.1f}% "
                  f"{metrics.durability_score:>10.1f}% "
                  f"{metrics.overall_score:>7.1f}%")
        
        print("\n" + "="*60)
        
        # Print best in category
        comparison = self._generate_comparison()
        print("\nBest in Category:")
        for category, db in comparison['best_in_category'].items():
            print(f"  {category.capitalize()}: {db}")
        
        print("="*60 + "\n")