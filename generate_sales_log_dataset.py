import csv
import random
from faker import Faker
from datetime import datetime, timedelta
import uuid

# Initialize Faker
fake = Faker()

# Define constants
COUNTRIES = [fake.country() for _ in range(200)]
SERVICES = ['Consulting', 'Support', 'Training', 'Implementation', 'Customization']
PRODUCT_TYPES = ['Software', 'Hardware', 'Subscription', 'Service']
BROWSERS = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
REQUEST_CATEGORIES = ['Product Page', 'Demo Request', 'Service Inquiry', 'Promotional Request', 'Virtual Assistant']
CUSTOMER_BEHAVIORS = ['Browsing', 'Engaged', 'Converted', 'Abandoned']
SUBSCRIPTION_STATUSES = ['Active', 'Trial', 'Expired', 'Cancelled']
CUSTOMER_SEGMENTS = ['Enterprise', 'SMB', 'Individual', 'Government']
GENDERS = ['Male', 'Female', 'Other']
JOB_TYPES = ['Data Analysis', 'Software Development', 'Consulting', 'Training', 'Support']  # Added for types_of_jobs_requested

# Regional profiles for country-specific variations
REGIONS = {
    'North America': {'countries': ['United States', 'Canada', 'Mexico'], 'revenue_range': (0, 10000), 'demo_prob': 0.3, 'session_range': (60, 5400), 'behavior_weights': [0.2, 0.3, 0.4, 0.1], 'request_weights': [0.3, 0.2, 0.2, 0.2, 0.1]},
    'Europe': {'countries': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain'], 'revenue_range': (0, 8000), 'demo_prob': 0.25, 'session_range': (60, 7200), 'behavior_weights': [0.25, 0.35, 0.3, 0.1], 'request_weights': [0.25, 0.25, 0.3, 0.15, 0.05]},
    'Asia': {'countries': ['China', 'Japan', 'India', 'South Korea', 'Singapore'], 'revenue_range': (0, 5000), 'demo_prob': 0.4, 'session_range': (30, 3600), 'behavior_weights': [0.3, 0.3, 0.25, 0.15], 'request_weights': [0.2, 0.4, 0.2, 0.1, 0.1]},
    'South America': {'countries': ['Brazil', 'Argentina', 'Colombia'], 'revenue_range': (0, 3000), 'demo_prob': 0.1, 'session_range': (30, 2400), 'behavior_weights': [0.4, 0.3, 0.15, 0.15], 'request_weights': [0.3, 0.1, 0.4, 0.1, 0.1]},
    'Africa': {'countries': ['Nigeria', 'South Africa', 'Kenya'], 'revenue_range': (0, 2000), 'demo_prob': 0.15, 'session_range': (30, 1800), 'behavior_weights': [0.5, 0.3, 0.1, 0.1], 'request_weights': [0.4, 0.15, 0.3, 0.1, 0.05]},
    'Oceania': {'countries': ['Australia', 'New Zealand'], 'revenue_range': (0, 6000), 'demo_prob': 0.2, 'session_range': (60, 4800), 'behavior_weights': [0.3, 0.3, 0.3, 0.1], 'request_weights': [0.3, 0.2, 0.2, 0.2, 0.1]}
}

def get_region(country):
    for region, data in REGIONS.items():
        if country in data['countries']:
            return region
    return 'Other'

# Generate synthetic log entry
def generate_log_entry():
    timestamp = fake.date_time_between(start_date='-30d', end_date='now')
    country = random.choice(COUNTRIES)
    region = get_region(country)
    
    # Default values for 'Other' regions
    revenue_range = (0, 4000)
    demo_prob = 0.2
    session_range = (30, 3600)
    behavior_weights = [0.3, 0.3, 0.2, 0.2]
    request_weights = [0.3, 0.2, 0.3, 0.1, 0.1]
    
    # Apply region-specific variations
    if region != 'Other':
        region_data = REGIONS[region]
        revenue_range = region_data['revenue_range']
        demo_prob = region_data['demo_prob']
        session_range = region_data['session_range']
        behavior_weights = region_data['behavior_weights']
        request_weights = region_data['request_weights']
    
    session_duration = random.randint(session_range[0], session_range[1])
    revenue = round(random.uniform(revenue_range[0], revenue_range[1]), 2) if random.choice([True, False]) else 0
    num_sales = random.randint(0, 5) if revenue > 0 else 0
    conversion_status = 'Converted' if num_sales > 0 else random.choice(['Not Converted', 'Pending'])
    demo_request = random.random() < demo_prob
    # Added fields
    num_jobs = random.randint(1, 10) if revenue > 0 else random.randint(0, 3)
    job_types = ','.join(random.sample(JOB_TYPES, k=random.randint(1, len(JOB_TYPES)))) if num_jobs > 0 else ''

    return {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'country': country,
        'service_type': random.choice(SERVICES),
        'demo_request': demo_request,
        'promotional_request': random.choice([True, False]),
        'virtual_assistant_request': random.choice([True, False]),
        'customer_behavior': random.choices(CUSTOMER_BEHAVIORS, weights=behavior_weights, k=1)[0],
        'marketing_performance': random.randint(1, 100),
        'product_interest': random.randint(1, 100),
        'user_journey': '/'.join(fake.words(nb=random.randint(2, 5))),
        'subscription_status': random.choice(SUBSCRIPTION_STATUSES),
        'revenue': revenue,
        'customer_segment': random.choice(CUSTOMER_SEGMENTS),
        'request_url': fake.uri(),
        'request_category': random.choices(REQUEST_CATEGORIES, weights=request_weights, k=1)[0],
        'product_id': f'PROD-{random.randint(1000, 9999)}',
        'browser': random.choice(BROWSERS),
        'product_type': random.choice(PRODUCT_TYPES),
        'transaction_id': str(uuid.uuid4()),
        'session_duration': session_duration,
        'expense_statement': round(random.uniform(0, 1000), 2),
        'number_of_sales': num_sales,
        'conversion_status': conversion_status,
        'gender': random.choice(GENDERS),
        'number_of_jobs_placed': num_jobs,  # Added
        'types_of_jobs_requested': job_types  # Added
    }

# Generate dataset
num_records = 10000  # Increased for better variation
log_entries = [generate_log_entry() for _ in range(num_records)]

# Define CSV headers
headers = [
    'timestamp', 'country', 'service_type', 'demo_request', 'promotional_request',
    'virtual_assistant_request', 'customer_behavior', 'marketing_performance',
    'product_interest', 'user_journey', 'subscription_status', 'revenue',
    'customer_segment', 'request_url', 'request_category', 'product_id',
    'browser', 'product_type', 'transaction_id', 'session_duration',
    'expense_statement', 'number_of_sales', 'conversion_status', 'gender',
    'number_of_jobs_placed', 'types_of_jobs_requested'  # Added
]

# Write to CSV
with open('sales_log_dataset.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(log_entries)

print(f"Generated {num_records} synthetic log entries in 'sales_log_dataset.csv'")