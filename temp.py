from pathlib import Path

import pandas as pd

name = 'www-vanartgallery-bc-ca-.png'
all_metrics = pd.read_csv("integer.csv", delimiter=',')
screenshots_of_domain = all_metrics[all_metrics.domain == 'food']
#rslt_df = all_metrics.loc[all_metrics['filename'] == name]['Aesthetics'].values[0]


available_screenshots = [(p.name, p) for p in Path('/home/bakaev/leonid/images2').glob('*/*.png')]

missing_samples = all_metrics[
    ~all_metrics.filename.isin([name for name, _ in available_screenshots])]

print(missing_samples)