from google_play_scraper import app, reviews_all, Sort
import pandas as pd

scrapreview = reviews_all(
    'superapps.polri.presisi.presisi',
    lang='id',
    country='id',
    sort=Sort.MOST_RELEVANT,
    count=10000
)

df = pd.DataFrame(scrapreview)
df.to_csv('app_reviews.csv', index=False, encoding='utf-8-sig')