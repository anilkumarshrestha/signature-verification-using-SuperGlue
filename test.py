#!/usr/bin/env python3
"""
Automatic Dataset Testing Script for Signature Verification
Dataset: https://www.kaggle.com/datasets/robinreni/signature-verification-dataset/data
Tests all signatures in a dataset with the following structure:
- dataset_path/001/ -> genuine signatures
- dataset_path/001_forg/ -> forged signatures
- dataset_path/002/ -> genuine signatures
- dataset_path/002_forg/ -> forged signatures
etc.
"""

import cv2
import torch
import numpy as np
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from signature_analysis import analyze_signatures_with_rotation, create_visualization, add_text_overlay, load_superglue_model

class SignatureDatasetTester:
    def __init__(self, dataset_path, output_dir="test_results", use_rotation=True, use_preprocessing=True):
        """
        Initialize the dataset tester

        Args:
            dataset_path: Path to dataset root directory
            output_dir: Directory to save results
            use_rotation: Enable rotation analysis
            use_preprocessing: Enable K-means preprocessing
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.use_rotation = use_rotation
        self.use_preprocessing = use_preprocessing

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)

        # Load model
        print("Loading SuperGlue model...")
        self.matching, self.device = load_superglue_model()
        print(f"Model loaded successfully on {self.device}")

        # Results storage
        self.results = []
        self.statistics = {
            'genuine_genuine': {'tp': 0, 'fn': 0, 'total': 0, 'scores': []},
            'genuine_forged': {'tn': 0, 'fp': 0, 'total': 0, 'scores': []},
        }

    def find_person_folders(self):
        """Find all person folders in the dataset"""
        person_folders = []

        # Find all numeric folders (e.g., 001, 002, etc.)
        for folder in sorted(self.dataset_path.iterdir()):
            if folder.is_dir() and folder.name.isdigit():
                person_id = folder.name
                genuine_dir = self.dataset_path / person_id
                forged_dir = self.dataset_path / f"{person_id}_forg"

                if genuine_dir.exists() and forged_dir.exists():
                    person_folders.append({
                        'id': person_id,
                        'genuine_dir': genuine_dir,
                        'forged_dir': forged_dir
                    })

        return person_folders

    def load_image(self, image_path):
        """Load and convert image to grayscale"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def get_image_files(self, directory):
        """Get all image files from a directory"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []

        for ext in extensions:
            image_files.extend(list(directory.glob(ext)))
            image_files.extend(list(directory.glob(ext.upper())))

        return sorted(image_files)

    def test_pair(self, img1_path, img2_path, ground_truth_same):
        """Test a pair of signatures"""
        # Load images
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        # Resize if necessary
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Analyze signatures
        result = analyze_signatures_with_rotation(
            img1, img2, self.matching, self.device,
            base_threshold=0.25,
            rotation_threshold=0.45,
            rotation_improvement_threshold=0.08,
            use_rotation=self.use_rotation,
            use_preprocessing=self.use_preprocessing,
            preprocessing_method='kmeans'
        )

        # Add metadata
        result['image1_path'] = str(img1_path)
        result['image2_path'] = str(img2_path)
        result['ground_truth_same'] = ground_truth_same
        result['correct_prediction'] = result['predicted_same'] == ground_truth_same

        return result

    def save_visualization(self, result, person_id, test_type, idx):
        """Save visualization for a test result"""
        vis = create_visualization(result)
        vis = add_text_overlay(vis, result, result['ground_truth_same'])

        # Save with descriptive filename
        filename = f"{person_id}_{test_type}_{idx:03d}.jpg"
        filepath = self.vis_dir / filename
        cv2.imwrite(str(filepath), vis)

        return str(filepath)

    def test_person(self, person_info):
        """Test all signature combinations for one person"""
        person_id = person_info['id']
        genuine_dir = person_info['genuine_dir']
        forged_dir = person_info['forged_dir']

        # Get image files
        genuine_images = self.get_image_files(genuine_dir)
        forged_images = self.get_image_files(forged_dir)

        print(f"\nPerson {person_id}: {len(genuine_images)} genuine, {len(forged_images)} forged")

        person_results = {
            'person_id': person_id,
            'genuine_genuine_tests': [],
            'genuine_forged_tests': []
        }

        # Test 1: Genuine vs Genuine (should match)
        print(f"  Testing genuine vs genuine...")
        for i in range(len(genuine_images)):
            for j in range(i + 1, len(genuine_images)):
                try:
                    result = self.test_pair(genuine_images[i], genuine_images[j], ground_truth_same=True)

                    # Update statistics
                    if result['predicted_same']:
                        self.statistics['genuine_genuine']['tp'] += 1
                    else:
                        self.statistics['genuine_genuine']['fn'] += 1
                    self.statistics['genuine_genuine']['total'] += 1
                    self.statistics['genuine_genuine']['scores'].append(result['ratio'])

                    # Save visualization for some samples
                    if len(person_results['genuine_genuine_tests']) < 5:  # Save first 5
                        result['visualization_path'] = self.save_visualization(
                            result, person_id, "genuine_vs_genuine", len(person_results['genuine_genuine_tests'])
                        )

                    person_results['genuine_genuine_tests'].append({
                        'image1': genuine_images[i].name,
                        'image2': genuine_images[j].name,
                        'predicted_same': result['predicted_same'],
                        'ratio': result['ratio'],
                        'correct': result['correct_prediction']
                    })

                    self.results.append(result)
                except Exception as e:
                    print(f"    Error testing {genuine_images[i].name} vs {genuine_images[j].name}: {e}")

        # Test 2: Genuine vs Forged (should NOT match)
        print(f"  Testing genuine vs forged...")
        for i, genuine_img in enumerate(genuine_images):
            for j, forged_img in enumerate(forged_images):
                try:
                    result = self.test_pair(genuine_img, forged_img, ground_truth_same=False)

                    # Update statistics
                    if not result['predicted_same']:
                        self.statistics['genuine_forged']['tn'] += 1
                    else:
                        self.statistics['genuine_forged']['fp'] += 1
                    self.statistics['genuine_forged']['total'] += 1
                    self.statistics['genuine_forged']['scores'].append(result['ratio'])

                    # Save visualization for some samples
                    if len(person_results['genuine_forged_tests']) < 5:  # Save first 5
                        result['visualization_path'] = self.save_visualization(
                            result, person_id, "genuine_vs_forged", len(person_results['genuine_forged_tests'])
                        )

                    person_results['genuine_forged_tests'].append({
                        'genuine_image': genuine_img.name,
                        'forged_image': forged_img.name,
                        'predicted_same': result['predicted_same'],
                        'ratio': result['ratio'],
                        'correct': result['correct_prediction']
                    })

                    self.results.append(result)
                except Exception as e:
                    print(f"    Error testing {genuine_img.name} vs {forged_img.name}: {e}")

        return person_results

    def calculate_metrics(self):
        """Calculate overall performance metrics"""
        metrics = {}

        # Genuine vs Genuine metrics
        gg = self.statistics['genuine_genuine']
        if gg['total'] > 0:
            metrics['genuine_genuine'] = {
                'total_tests': gg['total'],
                'true_positives': gg['tp'],
                'false_negatives': gg['fn'],
                'accuracy': gg['tp'] / gg['total'],
                'avg_score': np.mean(gg['scores']) if gg['scores'] else 0,
                'std_score': np.std(gg['scores']) if gg['scores'] else 0,
            }

        # Genuine vs Forged metrics
        gf = self.statistics['genuine_forged']
        if gf['total'] > 0:
            metrics['genuine_forged'] = {
                'total_tests': gf['total'],
                'true_negatives': gf['tn'],
                'false_positives': gf['fp'],
                'accuracy': gf['tn'] / gf['total'],
                'avg_score': np.mean(gf['scores']) if gf['scores'] else 0,
                'std_score': np.std(gf['scores']) if gf['scores'] else 0,
            }

        # Overall metrics
        total_correct = gg['tp'] + gf['tn']
        total_tests = gg['total'] + gf['total']

        if total_tests > 0:
            metrics['overall'] = {
                'total_tests': total_tests,
                'correct_predictions': total_correct,
                'overall_accuracy': total_correct / total_tests,
                'precision': gg['tp'] / (gg['tp'] + gf['fp']) if (gg['tp'] + gf['fp']) > 0 else 0,
                'recall': gg['tp'] / (gg['tp'] + gg['fn']) if (gg['tp'] + gg['fn']) > 0 else 0,
            }

            # F1 Score
            p = metrics['overall']['precision']
            r = metrics['overall']['recall']
            metrics['overall']['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        return metrics

    def plot_results(self, metrics):
        """Create visualization plots"""
        # Set style
        sns.set_style("whitegrid")

        # 1. Score distributions
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Genuine vs Genuine scores
        if self.statistics['genuine_genuine']['scores']:
            axes[0].hist(self.statistics['genuine_genuine']['scores'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0].axvline(0.25, color='red', linestyle='--', label='Base Threshold (0.25)')
            axes[0].set_xlabel('Match Ratio')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Genuine vs Genuine - Match Score Distribution')
            axes[0].legend()

        # Genuine vs Forged scores
        if self.statistics['genuine_forged']['scores']:
            axes[1].hist(self.statistics['genuine_forged']['scores'], bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1].axvline(0.25, color='red', linestyle='--', label='Base Threshold (0.25)')
            axes[1].set_xlabel('Match Ratio')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Genuine vs Forged - Match Score Distribution')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distributions.png', dpi=300)
        plt.close()

        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = np.array([
            [self.statistics['genuine_genuine']['tp'], self.statistics['genuine_genuine']['fn']],
            [self.statistics['genuine_forged']['fp'], self.statistics['genuine_forged']['tn']]
        ])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Same', 'Predicted Different'],
                    yticklabels=['Actually Same', 'Actually Different'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Ground Truth')
        ax.set_xlabel('Prediction')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()

        # 3. Metrics bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_data = {
            'Accuracy': metrics['overall']['overall_accuracy'],
            'Precision': metrics['overall']['precision'],
            'Recall': metrics['overall']['recall'],
            'F1 Score': metrics['overall']['f1_score']
        }

        bars = ax.bar(metrics_data.keys(), metrics_data.values(), color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Metrics')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
        ax.legend()

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_summary.png', dpi=300)
        plt.close()

    def save_report(self, metrics, person_results_list):
        """Save detailed report"""
        report = {
            'test_info': {
                'dataset_path': str(self.dataset_path),
                'timestamp': datetime.now().isoformat(),
                'rotation_enabled': self.use_rotation,
                'preprocessing_enabled': self.use_preprocessing,
                'device': self.device
            },
            'metrics': metrics,
            'person_results': person_results_list
        }

        # Save JSON report
        with open(self.output_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Save human-readable report
        with open(self.output_dir / 'report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SIGNATURE VERIFICATION TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Rotation: {'Enabled' if self.use_rotation else 'Disabled'}\n")
            f.write(f"Preprocessing: {'Enabled (K-means)' if self.use_preprocessing else 'Disabled'}\n")
            f.write(f"Device: {self.device.upper()}\n\n")

            f.write("=" * 80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("=" * 80 + "\n\n")

            om = metrics['overall']
            f.write(f"Total Tests: {om['total_tests']}\n")
            f.write(f"Correct Predictions: {om['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {om['overall_accuracy']:.4f} ({om['overall_accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {om['precision']:.4f}\n")
            f.write(f"Recall: {om['recall']:.4f}\n")
            f.write(f"F1 Score: {om['f1_score']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("GENUINE vs GENUINE TESTS\n")
            f.write("=" * 80 + "\n\n")

            gg = metrics['genuine_genuine']
            f.write(f"Total Tests: {gg['total_tests']}\n")
            f.write(f"True Positives (Correctly identified as same): {gg['true_positives']}\n")
            f.write(f"False Negatives (Incorrectly identified as different): {gg['false_negatives']}\n")
            f.write(f"Accuracy: {gg['accuracy']:.4f} ({gg['accuracy']*100:.2f}%)\n")
            f.write(f"Average Score: {gg['avg_score']:.4f} ± {gg['std_score']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("GENUINE vs FORGED TESTS\n")
            f.write("=" * 80 + "\n\n")

            gf = metrics['genuine_forged']
            f.write(f"Total Tests: {gf['total_tests']}\n")
            f.write(f"True Negatives (Correctly identified as different): {gf['true_negatives']}\n")
            f.write(f"False Positives (Incorrectly identified as same): {gf['false_positives']}\n")
            f.write(f"Accuracy: {gf['accuracy']:.4f} ({gf['accuracy']*100:.2f}%)\n")
            f.write(f"Average Score: {gf['avg_score']:.4f} ± {gf['std_score']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("PER-PERSON RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for person_result in person_results_list:
                person_id = person_result['person_id']
                gg_tests = person_result['genuine_genuine_tests']
                gf_tests = person_result['genuine_forged_tests']

                f.write(f"Person {person_id}:\n")
                f.write(f"  Genuine vs Genuine: {len(gg_tests)} tests\n")
                if gg_tests:
                    gg_correct = sum(1 for t in gg_tests if t['correct'])
                    f.write(f"    Accuracy: {gg_correct}/{len(gg_tests)} ({gg_correct/len(gg_tests)*100:.1f}%)\n")

                f.write(f"  Genuine vs Forged: {len(gf_tests)} tests\n")
                if gf_tests:
                    gf_correct = sum(1 for t in gf_tests if t['correct'])
                    f.write(f"    Accuracy: {gf_correct}/{len(gf_tests)} ({gf_correct/len(gf_tests)*100:.1f}%)\n")
                f.write("\n")

        print(f"\nReport saved to: {self.output_dir / 'report.txt'}")

    def run(self):
        """Run the complete test suite"""
        print(f"\n{'='*80}")
        print("Starting Signature Verification Dataset Testing")
        print(f"{'='*80}\n")

        # Find all person folders
        person_folders = self.find_person_folders()
        print(f"Found {len(person_folders)} persons to test")

        if not person_folders:
            print("No person folders found! Expected format: 001/, 001_forg/, 002/, 002_forg/, etc.")
            return

        # Test each person
        person_results_list = []
        for person_info in tqdm(person_folders, desc="Testing persons"):
            person_result = self.test_person(person_info)
            person_results_list.append(person_result)

        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = self.calculate_metrics()

        # Create plots
        print("Creating visualizations...")
        self.plot_results(metrics)

        # Save report
        print("Saving report...")
        self.save_report(metrics, person_results_list)

        # Print summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}\n")
        print(f"Total Tests: {metrics['overall']['total_tests']}")
        print(f"Overall Accuracy: {metrics['overall']['overall_accuracy']:.4f} ({metrics['overall']['overall_accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall: {metrics['overall']['recall']:.4f}")
        print(f"F1 Score: {metrics['overall']['f1_score']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Automatic Signature Dataset Testing')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results (default: test_results)')
    parser.add_argument('--no-rotation', action='store_true',
                       help='Disable rotation analysis')
    parser.add_argument('--no-preprocessing', action='store_true',
                       help='Disable K-means preprocessing')

    args = parser.parse_args()

    # Create tester
    tester = SignatureDatasetTester(
        dataset_path=args.dataset_path,
        output_dir=args.output,
        use_rotation=not args.no_rotation,
        use_preprocessing=not args.no_preprocessing
    )

    # Run tests
    tester.run()


if __name__ == '__main__':
    main()
