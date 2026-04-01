import pandas as pd
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
postings = pd.read_csv("data/raw/postings.csv")
job_skills = pd.read_csv("data/raw/jobs/job_skills.csv")
skills_map = pd.read_csv("data/raw/mappings/skills.csv")
industries = pd.read_csv("data/raw/jobs/job_industries.csv")

# Basic shapes
print(f"\nPostings: {postings.shape}")
print(f"Job skills pairs: {job_skills.shape}")
print(f"Unique jobs with skill labels: {job_skills['job_id'].nunique()}")
print(f"Unique skills: {job_skills['skill_abr'].nunique()}")

# How many labeled jobs also have descriptions?
labeled_jobs = set(job_skills['job_id'].unique())
postings_with_desc = postings[postings['description'].notna()]
overlap = postings_with_desc[postings_with_desc['job_id'].isin(labeled_jobs)]
print(f"\nPostings with description: {len(postings_with_desc)}")
print(f"Labeled jobs with description (usable): {len(overlap)}")

# Skills per job distribution
skills_per_job = job_skills.groupby('job_id')['skill_abr'].count()
print(f"\nSkills per job — mean: {skills_per_job.mean():.1f}, "
      f"median: {skills_per_job.median():.1f}, "
      f"max: {skills_per_job.max()}")

# Description length
overlap['desc_len'] = overlap['description'].str.len()
print(f"\nDescription length (chars) — mean: {overlap['desc_len'].mean():.0f}, "
      f"median: {overlap['desc_len'].median():.0f}")

# Skills legend
print(f"\nAll skill categories:\n{skills_map.to_string()}")