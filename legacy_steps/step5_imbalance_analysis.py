"""
PF-MDD Class Imbalance Analysis and Solutions
Comprehensive analysis of class distribution and mitigation strategies
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from project_config import DATA_DIR, preferred_training_dataset_dir

sys.stdout.reconfigure(encoding='utf-8')

class ImbalanceAnalyzer:
    def __init__(self, data_root: str, dataset_dir: str | Path | None = None):
        self.data_root = Path(data_root)
        self.dataset_dir = Path(dataset_dir) if dataset_dir else preferred_training_dataset_dir()
        self.analysis_results = {
            'dataset_overview': {},
            'class_distribution': {},
            'imbalance_metrics': {},
            'impact_analysis': {},
            'recommended_solutions': [],
            'implementation_priority': []
        }
    
    def load_statistics(self):
        """Load all available statistics from previous steps"""
        print("\n" + "="*70)
        print("[STEP 1] Loading Dataset Statistics")
        print("="*70)
        
        stats_files = [
            ('dataset_info', self.data_root / "pf_mdd_dataset_info.json"),
            ('conversion_stats', self.data_root / "yolo_labels" / "conversion_stats.json"),
            ('augmentation_stats', self.data_root / "augmented_dataset" / "augmentation_stats.json"),
            ('split_stats', self.dataset_dir / "split_statistics.json"),
            ('class_balance_summary', self.dataset_dir / "class_balance_summary.json"),
            ('expansion_summary', self.dataset_dir / "expansion_summary.json"),
        ]
        
        loaded_data = {}
        
        for name, path in stats_files:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    loaded_data[name] = json.load(f)
                print(f"[OK] Loaded {name}")
            else:
                print(f"[SKIP] {name} not found at {path}")
        
        return loaded_data

    def _count_boxes_from_train_labels(self) -> dict[int, int]:
        """Fallback class-count loader based on current train labels."""
        label_dir = self.dataset_dir / "labels" / "train"
        class_counts = defaultdict(int)
        if not label_dir.exists():
            return {}

        for label_path in sorted(label_dir.glob("*.txt")):
            try:
                for line in label_path.read_text(encoding='utf-8').splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    class_counts[class_id] += 1
            except Exception:
                continue
        return dict(class_counts)
    
    def analyze_class_distribution(self, loaded_data: dict):
        """Analyze the distribution of classes in detail"""
        print("\n" + "="*70)
        print("[STEP 2] Analyzing Class Distribution")
        print("="*70)
        
        if 'conversion_stats' in loaded_data:
            boxes_per_class = loaded_data['conversion_stats'].get('boxes_per_class', {})
            
            if isinstance(boxes_per_class, dict):
                class_counts = {int(k): v for k, v in boxes_per_class.items()}
            else:
                class_counts = {}
        else:
            class_counts = {}

        if not class_counts:
            class_counts = self._count_boxes_from_train_labels()
            if class_counts:
                print(f"[INFO] Loaded class distribution from train labels: {self.dataset_dir / 'labels' / 'train'}")
        
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        if total_samples == 0 or num_classes == 0:
            print("[WARNING] No class data available for analysis")
            return
        
        mean_count = np.mean(list(class_counts.values()))
        std_count = np.std(list(class_counts.values()))
        median_count = np.median(list(class_counts.values()))
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        cv = (std_count / mean_count * 100) if mean_count > 0 else 0
        
        gini_coefficient = self._calculate_gini(list(class_counts.values()))
        
        entropy = self._calculate_entropy(list(class_counts.values()), total_samples)
        max_entropy = np.log(num_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        print(f"\n[Dataset Overview]")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of classes: {num_classes}")
        print(f"\n[Distribution Statistics]")
        print(f"  Mean samples per class: {mean_count:.1f}")
        print(f"  Standard deviation: {std_count:.1f}")
        print(f"  Median: {median_count:.1f}")
        print(f"  Min: {min_count} (Class {min(class_counts, key=class_counts.get)})")
        print(f"  Max: {max_count} (Class {max(class_counts, key=class_counts.get)})")
        
        print(f"\n[Imbalance Metrics]")
        print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}:1")
        print(f"  Coefficient of Variation: {cv:.1f}%")
        print(f"  Gini Coefficient: {gini_coefficient:.4f}")
        print(f"  Entropy: {entropy:.4f} (Normalized: {normalized_entropy:.4f})")
        
        print(f"\n[Detailed Class Distribution]")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (cls_id, count) in enumerate(sorted_classes, 1):
            percentage = count / total_samples * 100
            bar_length = int(count / max_count * 50)
            bar = "#" * bar_length
            
            status = ""
            if count < mean_count * 0.3:
                status = "[SEVERE UNDER-REPRESENTED]"
            elif count < mean_count * 0.7:
                status = "[UNDER-REPRESENTED]"
            elif count > mean_count * 1.5:
                status = "[OVER-REPRESENTED]"
            else:
                status = "[BALANCED]"
            
            print(f"  {rank:2d}. Class {cls_id:03d}: {count:4d} ({percentage:5.1f}%) {bar} {status}")
        
        self.analysis_results['dataset_overview'] = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'mean': mean_count,
            'std': std_count,
            'median': median_count,
            'min': min_count,
            'max': max_count
        }
        
        self.analysis_results['class_distribution'] = class_counts
        
        self.analysis_results['imbalance_metrics'] = {
            'imbalance_ratio': imbalance_ratio,
            'coefficient_of_variation': cv,
            'gini_coefficient': gini_coefficient,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy
        }
    
    def _calculate_gini(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        values = sorted(values)
        n = len(values)
        if n == 0:
            return 0
        
        cumsum = np.cumsum(values, dtype=float)
        gini = (2 * sum((i + 1) * v for i, v in enumerate(values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        return gini
    
    def _calculate_entropy(self, counts, total):
        """Calculate entropy of distribution"""
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log(p) for p in probs)
    
    def assess_impact(self):
        """Assess potential impact of class imbalance on model performance"""
        print("\n" + "="*70)
        print("[STEP 3] Assessing Impact on Model Performance")
        print("="*70)
        
        metrics = self.analysis_results['imbalance_metrics']
        ratio = metrics.get('imbalance_ratio', 1)
        gini = metrics.get('gini_coefficient', 0)
        cv = metrics.get('coefficient_of_variation', 0)
        
        severity_levels = {
            'mild': {'ratio_range': (1, 3), 'gini_range': (0, 0.25), 'cv_range': (0, 50)},
            'moderate': {'ratio_range': (3, 10), 'gini_range': (0.25, 0.45), 'cv_range': (50, 100)},
            'severe': {'ratio_range': (10, float('inf')), 'gini_range': (0.45, 1), 'cv_range': (100, float('inf'))}
        }
        
        severity_score = 0
        if ratio > 10:
            severity_score += 3
        elif ratio > 3:
            severity_score += 2
        elif ratio > 1.5:
            severity_score += 1
        
        if gini > 0.45:
            severity_score += 3
        elif gini > 0.25:
            severity_score += 2
        elif gini > 0.15:
            severity_score += 1
        
        if cv > 100:
            severity_score += 2
        elif cv > 50:
            severity_score += 1
        
        if severity_score <= 2:
            severity = "MILD"
            risk_level = "LOW"
            color = "[GREEN]"
        elif severity_score <= 5:
            severity = "MODERATE"
            risk_level = "MEDIUM"
            color = "[YELLOW]"
        else:
            severity = "SEVERE"
            risk_level = "HIGH"
            color = "[RED]"
        
        print(f"\n[Imbalance Severity Assessment]")
        print(f"  Overall Severity: {color} {severity}")
        print(f"  Risk Level: {risk_level}")
        print(f"  Severity Score: {severity_score}/8")
        
        print(f"\n[Potential Impacts]")
        
        impacts = []
        
        if ratio > 5:
            impacts.append({
                'issue': 'Bias toward majority classes',
                'effect': 'Model will likely perform poorly on rare defect types',
                'severity': 'HIGH'
            })
        
        if gini > 0.35:
            impacts.append({
                'issue': 'Skewed decision boundaries',
                'effect': 'Classification thresholds may be biased',
                'severity': 'MEDIUM'
            })
        
        if cv > 60:
            impacts.append({
                'issue': 'Unstable training dynamics',
                'effect': 'Loss function dominated by frequent classes',
                'severity': 'MEDIUM'
            })
        
        impacts.extend([
            {
                'issue': 'Misleading accuracy metric',
                'effect': 'High overall accuracy but poor performance on minority classes',
                'severity': 'HIGH'
            },
            {
                'issue': 'Poor generalization to rare defects',
                'effect': 'Model fails to detect uncommon but critical defect types',
                'severity': 'CRITICAL'
            },
            {
                'issue': 'Calibration issues',
                'effect': 'Probability estimates unreliable for imbalanced data',
                'severity': 'MEDIUM'
            }
        ])
        
        for idx, impact in enumerate(impacts, 1):
            print(f"\n  {idx}. {impact['issue']}")
            print(f"     Effect: {impact['effect']}")
            print(f"     Severity: {impact['severity']}")
        
        self.analysis_results['impact_analysis'] = {
            'severity': severity,
            'risk_level': risk_level,
            'severity_score': severity_score,
            'impacts': impacts
        }
    
    def recommend_solutions(self):
        """Recommend solutions based on analysis results"""
        print("\n" + "="*70)
        print("[STEP 4] Recommended Solutions")
        print("="*70)
        
        metrics = self.analysis_results['imbalance_metrics']
        severity = self.analysis_results['impact_analysis'].get('severity', 'MODERATE')
        
        solutions = []
        
        print(f"\n[SOLUTION 1: Data-Level Approaches]")
        print("-" * 50)
        
        sol1 = {
            'name': 'Oversampling Minority Classes',
            'priority': 'HIGH' if severity in ['MODERATE', 'SEVERE'] else 'MEDIUM',
            'methods': [
                'Random Oversampling (ROS)',
                'SMOTE (Synthetic Minority Over-sampling Technique)',
                'ADASYN (Adaptive Synthetic Sampling)',
                'Image-specific augmentations (already implemented)'
            ],
            'pros': ['Simple to implement', 'No information loss', 'Effective for moderate imbalance'],
            'cons': ['Can cause overfitting', 'Increases training time'],
            'recommended_for': ['Classes 27, 29, 36 (<6 samples)', 'Classes 16, 23, 25 (<30 samples)'],
            'implementation': '''
# Example: SMOTE-like augmentation for images
def smote_augmentation(image, mask, alpha=0.3):
    # Apply geometric transformations with interpolation
    # Combine features from similar samples
    augmented = apply_interpolation(image, alpha=alpha)
    return augmented
'''
        }
        solutions.append(sol1)
        
        print(f"  Name: {sol1['name']}")
        print(f"  Priority: {sol1['priority']}")
        print(f"  Methods:")
        for method in sol1['methods']:
            print(f"    - {method}")
        print(f"  Recommended for: {sol1['recommended_for']}")
        
        print(f"\n[SOLUTION 2: Algorithm-Level Approaches]")
        print("-" * 50)
        
        sol2 = {
            'name': 'Cost-Sensitive Learning',
            'priority': 'CRITICAL',
            'methods': [
                'Class-weighted loss functions',
                'Focal Loss (Lin et al., 2017)',
                'Balanced class weights in YOLO'
            ],
            'pros': ['No data modification needed', 'Directly addresses optimization objective'],
            'cons': ['Requires careful tuning', 'May slow convergence'],
            'implementation': '''
# Focal Loss implementation for defect detection
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

# For YOLO: Set class weights inversely proportional to frequency
class_weights = calculate_balanced_weights(class_counts)
'''
        }
        solutions.append(sol2)
        
        print(f"  Name: {sol2['name']}")
        print(f"  Priority: {sol2['priority']}")
        print(f"  Methods:")
        for method in sol2['methods']:
            print(f"    - {method}")
        
        print(f"\n[SOLUTION 3: Hybrid Approaches]")
        print("-" * 50)
        
        sol3 = {
            'name': 'Two-Phase Training Strategy',
            'priority': 'HIGH',
            'description': '''Phase 1: Train on balanced subset (undersample majority + oversample minority)
Phase 2: Fine-tune on full dataset with class weights''',
            'steps': [
                'Create balanced training set (~equal samples per class)',
                'Train model for initial epochs (e.g., 50-100 epochs)',
                'Switch to full dataset with weighted loss',
                'Fine-tune for additional epochs with lower learning rate'
            ],
            'expected_improvement': '+15-25% mAP on minority classes'
        }
        solutions.append(sol3)
        
        print(f"  Name: {sol3['name']}")
        print(f"  Priority: {sol3['priority']}")
        print(f"  Description: {sol3['description'][:100]}...")
        print(f"  Expected improvement: {sol3['expected_improvement']}")
        
        print(f"\n[SOLUTION 4: Evaluation Metrics]")
        print("-" * 50)
        
        sol4 = {
            'name': 'Appropriate Evaluation Metrics',
            'priority': 'CRITICAL',
            'metrics': [
                ('mAP (mean Average Precision)', 'Primary metric for object detection'),
                ('Per-class AP', 'Monitor individual class performance'),
                ('F1-Score (macro)', 'Balance precision and recall'),
                ('Confusion Matrix', 'Visualize per-class errors'),
                ('Precision-Recall Curves', 'Understand trade-offs')
            ],
            'warning': 'DO NOT rely solely on Accuracy - it is misleading for imbalanced datasets!'
        }
        solutions.append(sol4)
        
        print(f"  Name: {sol4['name']}")
        print(f"  Priority: {sol4['priority']}")
        print(f"  Recommended metrics:")
        for metric, desc in sol4['metrics']:
            print(f"    - {metric}: {desc}")
        print(f"\n  WARNING: {sol4['warning']}")
        
        print(f"\n[SOLUTION 5: Advanced Techniques]")
        print("-" * 50)
        
        sol5 = {
            'name': 'Ensemble and Specialized Methods',
            'priority': 'MEDIUM',
            'techniques': [
                'Easy-Hard Sample Mining (focus on difficult minority samples)',
                'Gradient Centralization (stabilize training)',
                'Test-Time Augmentation (TTA) for better inference)',
                'Multi-scale Training (detect defects at different scales)',
                'Semi-Supervised Learning (utilize unlabeled normal samples)'
            ],
            'best_for': 'Production systems requiring high reliability'
        }
        solutions.append(sol5)
        
        print(f"  Name: {sol5['name']}")
        print(f"  Priority: {sol5['priority']}")
        print(f"  Techniques:")
        for tech in sol5['techniques']:
            print(f"    - {tech}")
        
        self.analysis_results['recommended_solutions'] = solutions
        
        priority_order = [
            ('IMMEDIATE', ['Cost-Sensitive Learning', 'Appropriate Evaluation Metrics']),
            ('SHORT-TERM', ['Oversampling Minority Classes', 'Two-Phase Training']),
            ('LONG-TERM', ['Ensemble Methods', 'Advanced Techniques'])
        ]
        
        self.analysis_results['implementation_priority'] = priority_order
        
        print(f"\n[IMPLEMENTATION PRIORITY]")
        print("="*50)
        for phase, items in priority_order:
            print(f"\n{phase}:")
            for item in items:
                print(f"  → {item}")
    
    def generate_action_plan(self):
        """Generate concrete action plan"""
        print("\n" + "="*70)
        print("[STEP 5] Action Plan for PF-MDD Dataset")
        print("="*70)
        
        action_plan = """
╔══════════════════════════════════════════════════════════════╗
║           IMMEDIATE ACTIONS (This Week)                       ║
╠══════════════════════════════════════════════════════════════╣
║ ✓ Implement Focal Loss in YOLO model                         ║
║   - Set alpha=0.25, gamma=2.0                                ║
║   - Monitor loss curves for convergence                      ║
║                                                              ║
✓ Calculate balanced class weights                              ║
║   weight_i = N / (n_classes * count_i)                       ║
║   Apply to both classification and box regression losses      ║
║                                                              ║
✓ Update evaluation pipeline                                    ║
║   - Track mAP@0.5 and mAP@0.5:0.95                           ║
║   - Report per-class AP for all 12 categories                 ║
║   - Generate confusion matrix after each epoch               ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║          SHORT-TERM ACTIONS (Next 2-3 Weeks)                  ║
╠══════════════════════════════════════════════════════════════╣
║ ✓ Targeted augmentation for under-represented classes         ║
║   Priority classes:                                          ║
║   - CRITICAL: Classes 27, 29, 36 (only 1-6 samples each)     ║
║   - HIGH: Classes 16, 23, 25 (<30 samples)                   ║
║                                                              ║
║   Augmentation strategy:                                     ║
║   - Generate 20-30 synthetic samples per critical class       ║
║   - Use Mosaic, MixUp, CutMix combinations                   ║
║   - Apply domain-specific transforms (if known)              ║
║                                                              ║
✓ Implement two-phase training                                  ║
║   Phase 1 (Epochs 1-100): Balanced subset                     ║
║   Phase 2 (Epochs 101-150): Full dataset + weights            ║
║   Reduce LR by 10x at phase transition                       ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║           LONG-TERM OPTIMIZATION (Next Month)                  ║
╠══════════════════════════════════════════════════════════════╣
║ ✓ Ensemble multiple models                                   ║
║   - Train 3-5 models with different seeds/augmentations      ║
║   - Use weighted voting or NMS for predictions               ║
║                                                              ║
✓ Explore semi-supervised approaches                            ║
║   - Utilize 141 no-defect images as negative samples          ║
║   - Consider pseudo-labeling confident predictions            ║
║                                                              ║
✓ Production deployment considerations                          ║
║   - Set higher confidence threshold for rare classes          ║
║   - Implement human-in-the-loop verification                 ║
║   - Monitor drift in class distribution over time             ║
╚══════════════════════════════════════════════════════════════╝

📊 SUCCESS METRICS TO TRACK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Overall mAP target: ≥ 75% @ IoU 0.5
• Minimum per-class AP: ≥ 50% for all classes
• Critical classes (27, 29, 36): AP ≥ 40%
• Recall on minority classes: ≥ 70%
• False Positive Rate: ≤ 5% on normal samples
"""
        
        print(action_plan)
        
        report_file = self.data_root.parent / "PF_MDD_Imbalance_Analysis_Report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(action_plan)
        
        print(f"\n[SAVED] Full report saved to: {report_file}")

def main():
    data_root = str(DATA_DIR)
    
    print("\n" + "#"*70)
    print("# PF-MDD Class Imbalance Analysis Tool")
    print("# Comprehensive Analysis and Mitigation Strategies")
    print("#"*70)
    
    analyzer = ImbalanceAnalyzer(data_root)
    print(f"[INFO] Active dataset directory: {analyzer.dataset_dir}")
    
    loaded_data = analyzer.load_statistics()
    
    if loaded_data:
        analyzer.analyze_class_distribution(loaded_data)
        analyzer.assess_impact()
        analyzer.recommend_solutions()
        analyzer.generate_action_plan()
        
        print("\n" + "="*70)
        print("[SUCCESS] Imbalance analysis completed!")
        print("="*70)
    else:
        print("\n[ERROR] No statistics files found. Run previous steps first.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
