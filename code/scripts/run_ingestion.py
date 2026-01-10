"""Script to run document ingestion pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from code.src.ingestion.pipeline import IngestionPipeline
from code.src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/ingestion.log")

def main():
    """Run ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Process Sanskrit story files")
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory or file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed stories'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate statistics report'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IngestionPipeline(output_dir=args.output)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        logger.info(f"Processing single file: {input_path}")
        stories = pipeline.process_file(str(input_path))
    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        stories = pipeline.process_directory(str(input_path))
    else:
        logger.error(f"Invalid input path: {input_path}")
        return
    
    # Save stories
    if stories:
        pipeline.save_stories(stories)
        logger.info(f"Successfully processed {len(stories)} stories")
        
        # Generate report if requested
        if args.report:
            report = pipeline.generate_report(stories)
            report_path = Path(args.output) / "ingestion_report.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Report saved to {report_path}")
            
            # Print summary
            print("\n=== Ingestion Summary ===")
            print(f"Total stories: {report['total_stories']}")
            print(f"Total characters: {report['total_characters']}")
            print(f"Devanagari chars: {report['loader_stats']['devanagari_chars']}")
            print(f"Dandas: {report['loader_stats']['danda_count']}")
            print("\nStories:")
            for story in report['stories']:
                print(f"  {story['id']}. {story['title']} ({story['content_type']})")
    else:
        logger.warning("No stories were processed")

if __name__ == "__main__":
    main()