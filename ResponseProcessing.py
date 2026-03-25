#!/usr/bin/env python3
"""
Batch Results Consolidation Script

This script processes batch result files from Anthropic batch processing and consolidates
them into a single JSON file with image paths and processing statistics.

Features:
- Consolidates multiple batch result files
- Adds image path metadata to results
- Provides detailed processing statistics
- Supports multiple output directories
- Handles missing files gracefully
- Generates comprehensive reports

Author: LikelihoodRatio Project
Date: August 2025
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import sys

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.markdown import Markdown

console = Console()


class BatchResultsProcessor:
    """Main class for processing and consolidating batch results."""

    def __init__(self, target_dir: Path, output_file: Optional[Path] = None, ground_truth_csv: Optional[Path] = None):
        """
        Initialize the processor.

        Args:
            target_dir: Directory containing batch files
            output_file: Optional output file path
            ground_truth_csv: Optional path to CSV file with ground truth labels
        """
        self.target_dir = Path(target_dir)
        self.output_file = output_file or self.target_dir / 'consolidated_results.json'
        self.ground_truth_csv = ground_truth_csv
        self.stats = {
            'total_batches': 0,
            'total_results': 0,
            'successful_results': 0,
            'failed_results': 0,
            'missing_results': 0,
            'ground_truth_matches': 0,
            'processing_time': None
        }

    def validate_directory(self) -> bool:
        """Validate that the target directory exists and contains batch files."""
        if not self.target_dir.exists():
            console.print(f"[red]Error: Directory {self.target_dir} does not exist[/red]")
            return False

        info_files = list(self.target_dir.glob('batch_info_*.json'))
        result_files = list(self.target_dir.glob('batch_results_*.jsonl'))

        if not info_files:
            console.print(f"[red]Error: No batch_info_*.json files found in {self.target_dir}[/red]")
            return False

        if not result_files:
            console.print(f"[yellow]Warning: No batch_results_*.jsonl files found in {self.target_dir}[/yellow]")

        console.print(f"[green]✓ Found {len(info_files)} info files and {len(result_files)} result files[/green]")
        return True

    def get_batch_files(self) -> Tuple[List[Path], List[Path]]:
        """Get sorted lists of batch info and result files."""
        info_files = sorted(self.target_dir.glob('batch_info_*.json'))
        result_files = sorted(self.target_dir.glob('batch_results_*.jsonl'))

        self.stats['total_batches'] = len(info_files)
        return info_files, result_files

    def extract_batch_ids(self, info_files: List[Path]) -> set:
        """Extract unique batch IDs from info files."""
        batch_ids = set()
        for info_file in info_files:
            batch_id = info_file.stem.replace('batch_info_', '')
            batch_ids.add(batch_id)
        return batch_ids

    def load_pair_mapping(self, info_files: List[Path]) -> Dict[str, Tuple[str, str]]:
        """Load pair mapping from the first available info file."""
        pair_mapping = {}

        for info_file in info_files:
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)

                if 'pair_mapping' in info_data:
                    pair_mapping.update(info_data['pair_mapping'])

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                console.print(f"[yellow]Warning: Could not load pair mapping from {info_file.name}: {e}[/yellow]")
                continue

        console.print(f"[blue]✓ Loaded pair mapping for {len(pair_mapping)} image pairs[/blue]")
        return pair_mapping

    def load_ground_truth(self) -> Dict[Tuple[str, str], bool]:
        """Load ground truth labels from CSV file."""
        ground_truth = {}

        if not self.ground_truth_csv:
            return ground_truth

        csv_path = Path(self.ground_truth_csv)
        if not csv_path.exists():
            console.print(f"[yellow]Warning: Ground truth CSV file {csv_path} not found[/yellow]")
            return ground_truth

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Search the row with 'image1' and 'image2' columns
                for row in reader:
                    image1 = row.get('image1', '').strip()
                    image2 = row.get('image2', '').strip()
                    same_person = row.get('same_person', '').strip().lower() == 'true'

                    if image1 and image2:
                        key = tuple(sorted([image1, image2]))
                        ground_truth[key] = same_person

            console.print(f"[blue]✓ Loaded ground truth for {len(ground_truth)} image pairs[/blue]")

        except Exception as e:
            console.print(f"[red]Error loading ground truth CSV: {e}[/red]")

        return ground_truth

    def load_all_results(self, result_files: List[Path]) -> Dict[str, Any]:
        """Load and consolidate all results from result files."""
        all_results = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Loading results...", total=len(result_files))

            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue

                            try:
                                result = json.loads(line.strip())
                                custom_id = result.get('custom_id')

                                if custom_id:
                                    result_data = result.get('result', {})

                                    # Track success/failure
                                    if result_data.get('type') == 'succeeded':
                                        self.stats['successful_results'] += 1
                                    else:
                                        self.stats['failed_results'] += 1

                                    all_results[custom_id] = result_data
                                    self.stats['total_results'] += 1

                            except json.JSONDecodeError as e:
                                console.print(f"[red]Error parsing line {line_num} in {result_file.name}: {e}[/red]")

                except FileNotFoundError:
                    console.print(f"[red]Error: Could not read {result_file}[/red]")

                progress.advance(task)

        return all_results

    def add_image_metadata(self, all_results: Dict[str, Any], pair_mapping: Dict[str, Tuple[str, str]], ground_truth: Dict[Tuple[str, str], bool]) -> Dict[str, Any]:
        """Add image path metadata and ground truth labels to results."""
        enhanced_results = {}

        for custom_id in pair_mapping:
            image1, image2 = pair_mapping[custom_id]
            result = all_results.get(custom_id)

            if result is not None:
                # Add image paths to the result (last two parts only)
                result['image1'] = "/".join(image1.split("/")[-2:])
                result['image2'] = "/".join(image2.split("/")[-2:])
            enhanced_results[custom_id] = result

        return enhanced_results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save consolidated results to JSON file."""
        # Sort results by custom_id for consistency
        sorted_results = {k: results[k] for k in sorted(results.keys())}

        # Add metadata
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_directory': str(self.target_dir),
                'total_pairs': len(sorted_results),
                'statistics': self.stats
            },
            'results': sorted_results
        }

        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            console.print(f"[green]✓ Consolidated results written to {self.output_file}[/green]")

        except IOError as e:
            console.print(f"[red]Error writing output file: {e}[/red]")
            raise

    def display_statistics(self) -> None:
        """Display processing statistics in a nice table."""
        table = Table(title="Processing Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percentage", style="yellow", justify="right")

        total = self.stats['total_results'] + self.stats['missing_results']

        table.add_row("Total Batches", str(self.stats['total_batches']), "")
        table.add_row("Total Pairs", str(total), "100.0%")
        table.add_row("Successful Results", str(self.stats['successful_results']),
                      f"{self.stats['successful_results']/total*100:.1f}%" if total > 0 else "0.0%")
        table.add_row("Failed Results", str(self.stats['failed_results']),
                      f"{self.stats['failed_results']/total*100:.1f}%" if total > 0 else "0.0%")
        table.add_row("Missing Results", str(self.stats['missing_results']),
                      f"{self.stats['missing_results']/total*100:.1f}%" if total > 0 else "0.0%")
        table.add_row("Ground Truth Matches", str(self.stats['ground_truth_matches']),
                      f"{self.stats['ground_truth_matches']/total*100:.1f}%" if total > 0 else "0.0%")

        console.print(table)

    def process(self) -> bool:
        """Main processing function."""
        start_time = datetime.now()

        console.print("[blue]🚀 Starting batch results consolidation...[/blue]")

        # Validate directory
        if not self.validate_directory():
            return False

        # Get batch files
        info_files, result_files = self.get_batch_files()

        if not info_files:
            console.print("[red]No batch info files to process[/red]")
            return False

        # Load pair mapping
        pair_mapping = self.load_pair_mapping(info_files)
        if not pair_mapping:
            console.print("[red]Could not load any pair mappings[/red]")
            return False

        # Load all results
        all_results = self.load_all_results(result_files)

        # Load ground truth labels
        ground_truth = self.load_ground_truth()

        # Add image metadata and ground truth labels
        enhanced_results = self.add_image_metadata(all_results, pair_mapping, ground_truth)

        # Save results
        self.save_results(enhanced_results)

        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = str(end_time - start_time)

        # Display statistics
        self.display_statistics()

        console.print("[bold green]🎉 Batch results consolidation completed successfully![/bold green]")
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Consolidate batch processing results into a single JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default directory with default ground truth
  python ResponseProcessing.py
  
  # Process specific directory
  python ResponseProcessing.py --target-dir BatchFiles
  
  # Specify custom output file and ground truth CSV
  python ResponseProcessing.py --target-dir BatchFiles --output results.json --ground-truth my_pairs.csv
  
  # Process with custom ground truth file
  python ResponseProcessing.py --ground-truth custom_pairs.csv
        """
    )

    parser.add_argument("--target-dir", default="./BatchFiles(match)",
                        help="Directory containing batch files (default: ./BatchFiles(match))")
    parser.add_argument("--output", "-o",
                        help="Output JSON file path (default: <target_dir>/consolidated_results.json)")
    parser.add_argument("--ground-truth", "-g",
                        help="Path to CSV file with ground truth labels (default: pairs_balanced_with_filenames.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Setup paths
    target_dir = Path(args.target_dir)
    output_file = Path(args.output) if args.output else None
    ground_truth_csv = Path(args.ground_truth) if args.ground_truth else Path("pairs_balanced_with_filenames.csv")

    console.print(f"[blue]📁 Target directory:[/blue] {target_dir}")
    console.print(f"[blue]📁 Output file:[/blue] {output_file or target_dir / 'consolidated_results.json'}")
    console.print(f"[blue]📁 Ground truth CSV:[/blue] {ground_truth_csv}")

    try:
        # Create processor and run
        processor = BatchResultsProcessor(target_dir, output_file, ground_truth_csv)
        success = processor.process()

        return 0 if success else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
