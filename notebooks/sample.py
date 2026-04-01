import pandas as pd

print("Loading data...")
postings = pd.read_csv("data/raw/postings.csv")
job_skills = pd.read_csv("data/raw/jobs/job_skills.csv")
job_industries = pd.read_csv("data/raw/jobs/job_industries.csv")

# Only jobs that have descriptions AND skill labels
labeled_jobs = set(job_skills['job_id'].unique())
usable = postings[
    postings['description'].notna() &
    postings['job_id'].isin(labeled_jobs)
    ].copy()

# Sample 2000 jobs with a fixed seed for reproducibility
sample = usable.sample(n=2000, random_state=42)

# Keep only what we need
sample = sample[['job_id', 'title', 'company_name', 'description']].reset_index(drop=True)

# Attach ground truth skill labels
sample_skills = job_skills[job_skills['job_id'].isin(sample['job_id'])]
sample_industries = job_industries[job_industries['job_id'].isin(sample['job_id'])]

# Save
sample.to_csv("data/raw/sample_postings.csv", index=False)
sample_skills.to_csv("data/raw/sample_skills.csv", index=False)
sample_industries.to_csv("data/raw/sample_industries.csv", index=False)

print(f"Sample size: {len(sample)} jobs")
print(f"Sample skill pairs: {len(sample_skills)}")
print(f"Sample industry pairs: {len(sample_industries)}")
print(f"\nTitle examples:")
print(sample['title'].head(10).to_string())