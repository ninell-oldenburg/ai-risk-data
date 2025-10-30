import pandas as pd
import os
import glob
from typing import List
import nomquamgender as nqg
import time
import requests
import importlib
import chgender
from collections import Counter

class OpenAlexCSVProcessor:
    def __init__(self):
        self.input_dir = "src/processed_data/openalex/01_dedup"
        self.output_dir = "src/processed_data/openalex/02_with_gender"
        os.makedirs(self.output_dir, exist_ok=True)
        self.nqgmodel = nqg.NBGC()
        self.nqgmodel.threshold = .2
        self.GENDER_TERMS = {'male': 'gm', 'female': 'gf'}
        
    def process_all_csvs(self):
        csv_pattern = os.path.join(self.input_dir, "*", "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"No CSV files found in {self.input_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file}")
            try:
                self.process_single_csv(csv_file)
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
                continue
    
    def process_single_csv(self, filepath: str):
        df = pd.read_csv(filepath)
        print(f"  Loaded {len(df)} papers")

        df['author_genders'] = df.apply(
            lambda row: self.extract_first_author_gender(row['id'], row['author_names']), 
            axis=1
        )
        
        relative_path = os.path.relpath(filepath, self.input_dir)
        output_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved updated CSV: {output_path}")
    
    def extract_first_author_gender(self, paper_id: str, author_names: str) -> str:
        if pd.isna(author_names) or not author_names:
            return '-'
        
        authors = author_names.split(';')
        if not authors:
            return '-'
        
        author_genders = self.nqgmodel.classify(authors)
        for i, name in enumerate(authors):
            if author_genders[i] == '-':
                prediction, prob = chgender.guess(name)
                if prob > 0.8: 
                    author_genders[i] = self.GENDER_TERMS[prediction]
        
        author_genders_readable = ''
        for i, gender in enumerate(author_genders):
            author_genders_readable += gender
            if i+1 == len(author_genders):
                break
            author_genders_readable += '; '

        return author_genders_readable
    
    def generate_summary_report(self) -> pd.DataFrame:
        csv_pattern = os.path.join(self.output_dir, "*", "*.csv")
        csv_files = glob.glob(csv_pattern)

        male_first_author_count = 0
        female_first_author_count = 0
        unknown_first_author_count = 0
        total_paper_count = 0
        files_processed = 0

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                files_processed += 1
                
                first_author_genders = df['author_genders'].str.split(';').str[0].str.strip()
                
                male_first_author_count += (first_author_genders == 'gm').sum()
                female_first_author_count += (first_author_genders == 'gf').sum()
                unknown_first_author_count += (first_author_genders == '-').sum()
                total_paper_count += len(df)
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        summary_data = {
            'Metric': [
                'Files processed',
                'Total papers',
                'Male first author',
                'Female first author',
                'Unknown first author'
            ],
            'Count': [
                f"{files_processed}/{len(csv_files)}",
                total_paper_count,
                male_first_author_count,
                female_first_author_count,
                unknown_first_author_count
            ]
        }
        
        return pd.DataFrame(summary_data)

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = OpenAlexCSVProcessor()
    
    # Process all CSV files
    processor.process_all_csvs()
    
    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    summary = processor.generate_summary_report()
    print(summary.to_string())