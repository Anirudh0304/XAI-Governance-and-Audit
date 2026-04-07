import json
from datetime import datetime

master = {
    'last_updated': datetime.utcnow().isoformat() + 'Z',
    'datasets': [
        {
            'id': '1a',
            'name': 'German Credit',
            'sensitive': 'personal_status',
            'samples': 1000,
            'features_selected': 33,
            'accuracy': 0.760,
            'roc_auc': 0.765,
            'f1': 0.750,
            'oob_score': 0.829,
            'cv_auc': '0.787 +/- 0.037',
            'dp_diff': 0.113,
            'eo_diff': 0.333,
            'dp_ci': '[0.066 - 0.543]',
            'eo_ci': '[0.194 - 1.000]',
            'ci_reliable': False,
            'note': 'Wide CIs — small group sizes in personal_status'
        },
        {
            'id': '1b',
            'name': 'German Credit',
            'sensitive': 'age',
            'samples': 1000,
            'features_selected': 33,
            'accuracy': 0.760,
            'roc_auc': 0.765,
            'f1': 0.750,
            'oob_score': 0.829,
            'cv_auc': '0.787 +/- 0.037',
            'dp_diff': 0.110,
            'eo_diff': 0.286,
            'dp_ci': '[0.051 - 0.255]',
            'eo_ci': '[0.167 - 0.667]',
            'ci_reliable': False,
            'note': 'Wide CIs — senior group n=5 in test set'
        },
        {
            'id': '2',
            'name': 'Australian Credit',
            'sensitive': 'N/A — anonymised attributes',
            'samples': 690,
            'features_selected': 7,
            'accuracy': 0.855,
            'roc_auc': 0.934,
            'f1': 0.855,
            'oob_score': 0.866,
            'cv_auc': '0.931 +/- 0.022',
            'dp_diff': None,
            'eo_diff': None,
            'dp_ci': 'N/A',
            'eo_ci': 'N/A',
            'ci_reliable': False,
            'note': 'Anonymised attributes prevent meaningful fairness analysis'
        },
        {
            'id': '3',
            'name': 'Bank Marketing',
            'sensitive': 'marital',
            'samples': 45211,
            'features_selected': 27,
            'accuracy': 0.905,
            'roc_auc': 0.915,
            'f1': 0.905,
            'oob_score': 0.925,
            'cv_auc': '0.924 +/- 0.001',
            'dp_diff': 0.044,
            'eo_diff': 0.084,
            'dp_ci': '[0.030 - 0.060]',
            'eo_ci': '[0.033 - 0.167]',
            'ci_reliable': True,
            'note': 'Best fairness result so far — tight CIs, both metrics within limit'
        },
        {
            'id': '4',
            'name': 'Credit Risk Dataset',
            'sensitive': 'person_home_ownership',
            'samples': 28638,
            'features_selected': 'TBD',
            'accuracy': 0.908,
            'roc_auc': 0.921,
            'f1': 0.903,
            'dp_diff': 0.199,
            'eo_diff': 0.608,
            'dp_ci': '[0.184 - 0.233]',
            'eo_ci': '[0.526 - 0.858]',
            'ci_reliable': True,
            'note': 'DP at threshold boundary — confirmed genuine bias across home ownership groups'
        },
        {
            'id': '5',
            'name': 'Give Me Some Credit',
            'sensitive': 'age',
            'samples': 150000,
            'features_selected': 'TBD',
            'accuracy': 0.937,
            'roc_auc': 0.850,
            'f1': 0.918,
            'dp_diff': 0.030,
            'eo_diff': 0.191,
            'dp_ci': '[0.026 - 0.042]',
            'eo_ci': '[0.160 - 0.289]',
            'ci_reliable': True,
            'note': 'Very low DP diff — age groups treated fairly. EO within limit.'
        },
        {
            'id': '6',
            'name': 'Taiwan Credit Card Default',
            'sensitive': 'SEX',
            'samples': 30000,
            'features_selected': 'TBD',
            'accuracy': 0.812,
            'roc_auc': 0.764,
            'f1': 0.791,
            'dp_diff': 0.001,
            'eo_diff': 0.028,
            'dp_ci': '[0.000 - 0.018]',
            'eo_ci': '[0.004 - 0.079]',
            'ci_reliable': True,
            'note': 'Near-zero bias — lowest DP and EO across all datasets.'
        },
        {
            'id': '7',
            'name': 'HMDA Mortgage 2023',
            'sensitive': 'derived_race',
            'samples': 33263,
            'features_selected': 'TBD',
            'accuracy': 0.980,
            'roc_auc': 0.996,
            'f1': 0.980,
            'dp_diff': 0.133,
            'eo_diff': 0.139,
            'dp_ci': '[0.100 - 0.246]',
            'eo_ci': '[0.121 - 0.364]',
            'ci_reliable': True,
            'note': 'Near-perfect AUC. CI upper bounds cross 0.20 — borderline racial bias.'
        }
    ]
}

with open('reports/master_results.json', 'w') as f:
    json.dump(master, f, indent=2)

print('Master results recreated successfully')
print('Total datasets recorded:', len(master['datasets']))
print()
for d in master['datasets']:
    print(f"  {d['id']} — {d['name']} | AUC: {d['roc_auc']} | DP: {d['dp_diff']}")