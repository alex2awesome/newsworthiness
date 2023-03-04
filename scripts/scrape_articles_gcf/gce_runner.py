import requests
import json
import time
from tqdm.auto import tqdm
import jsonlines
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from random import choice

auth = open('gcf_auth.txt').read().strip()
urls = [
    'https://wayback-scrape-v2-2-ukvxfz3sya-uc.a.run.app',
    'https://wayback-scrape-v2-3-ukvxfz3sya-uw.a.run.app',
    'https://wayback-scrape-v2-5-ukvxfz3sya-ue.a.run.app'
]

def simple_gcf_wrapper(data):
    url = choice(urls)
    output = requests.post(
            url,
            headers={
                'Authorization': f'bearer {auth}',
                'Content-Type': 'application/json'
            },
            data=json.dumps(data)
        )
    if output.status_code == 200:
        if output.text == 'No items in Wayback Machine':
            print(output.text)
            return

        return output.json()
    else:
        print(output.status_code)
        if output.status_code == 500:
            print(str(data))
        return

#  --input-file ../data/latimes-article-urls-to-fetch.csv --output-file ../data/latimes-articles-8-years.jsonl --num-concurrent-workers 10

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', dest='input_file', type=str)
    parser.add_argument('--output-file', dest='output_file', type=str)
    parser.add_argument('--num-concurrent-workers', dest='workers', type=int)
    args = parser.parse_args()

    article_df = pd.read_csv(args.input_file, index_col=0)

    data = (
        article_df
            .rename(columns={'href': 'article_url', 'key': 'homepage_key'})
            .assign(homepage_key=lambda df: df['homepage_key'].astype(str))
            .to_dict(orient='records')
    )

    with open(args.output_file, 'w') as f:
        w = jsonlines.Writer(f)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for output in tqdm(executor.map(simple_gcf_wrapper, data), total=len(data)):
                if output is not None:
                    output.pop('article_html', None)
                    w.write(output)