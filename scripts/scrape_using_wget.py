import shutil

from pytimeparse.timeparse import timeparse
import os
from datetime import datetime
from page_exclusions import nytimes, ajc
from merge_two_dirs import main_merge as merge_dirs
from playwright.sync_api import sync_playwright
import subprocess
import glob
import requests

here = os.getcwd()
epoch_time = datetime(1970, 1, 1)
site_to_processor = {
    'nytimes.com': nytimes,
    'ajc.com': ajc,
}

def get_time_from_waybackpack_url(url_str:str):
    """
    Extracts the date string from a waybackurl and converts it to total_seconds from 1970/1/1
        :type url_str: str
    """
    date_str = url_str.split('/')[4]
    dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return (dt - epoch_time).total_seconds()


def rename_files_for_playwright(file_on_disk):
    fname = 'index.html'
    if not os.path.isdir(file_on_disk):
        temp_fname = os.path.join(os.path.dirname(file_on_disk), fname)
        perm_fname = os.path.join(file_on_disk, fname)
        os.rename(file_on_disk, temp_fname)
        os.mkdir(file_on_disk)
        shutil.move(temp_fname, perm_fname)
        file_on_disk = perm_fname
    else:
        file_on_disk = os.path.join(file_on_disk, fname)
        if not os.path.exists(file_on_disk):
            raise FileNotFoundError(f"{file_on_disk} doesn't exist!")
    return file_on_disk


if __name__ == '__main__':
    import argparse
    from subprocess import Popen, PIPE, run, check_call

    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default='nytimes.com')
    parser.add_argument('--from-date', dest='from_date', type=str, default='20230101')
    parser.add_argument('--to-date', dest='to_date', type=str, default='20230102')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='/dev/shm')
    parser.add_argument('--non-headless-browser', dest='headless', action='store_false')
    parser.add_argument('--verbose', action='store_true', )
    parser.add_argument(
        '--collapse',
        type=str,
        default='30m',
        help='Amount of time in between wayback page snapshots (e.g. 32m, 2h32m, 4:13, 1.2 minutes, 5hr34m56s.',
    )
    args = parser.parse_args()

    'http://web.archive.org/cdx/search/cdx?url=ajc.com&output=json&from=20230101&to=20230102'

    output, err = Popen([
        "waybackpack", args.site,
        "--from-date",
        args.from_date,
        "--to-date",
        args.to_date,
        "--list"
    ], stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()

    wayback_urls = output.decode().split()
    wayback_seconds = list(map(get_time_from_waybackpack_url, wayback_urls))

    collapse_time = timeparse(args.collapse)
    prev_datetime = wayback_seconds[0] - 2 * collapse_time  # default to make sure we get the first endpoint
    processor = site_to_processor.get(args.site)

    with sync_playwright() as p:
        # Open a browser
        browser = p.chromium.launch(channel="chrome", headless=args.headless)
        context = browser.new_context()
        page = context.new_page()
        page.route("**/*", lambda route: route.abort() if route.request.resource_type == "image" else route.continue_())
        page.route("https://web.archive.org*/*", lambda route: route.abort())

        for url, url_seconds in zip(wayback_urls, wayback_seconds):
            if (url_seconds - prev_datetime) < collapse_time:
                continue

            cmd = f'''wget \
                        -U 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36' \
                        --no-clobber \
                        --page-requisites \
                        --convert-links \
                        --timestamping \
                        --reject '*.ttf,*.woff2,*.woff,*.js,*.ico,*.txt,*.gif,*.jpg,*.jpeg,*.png,*.mp3,*.pdf,*.tgz,*.flv,*.avi,*.mpeg,*.iso' \
                        --ignore-tags=img \
                        --domains web.archive.org \
                        --no-parent {url} \
                        -P {os.path.join(here, 'tmp')}'''
            try:
                print(f'running wget on {url}...')
                if args.verbose:
                    wget_output = check_call(cmd, shell=True)
                else:
                    wget_output = check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                pass

            file_on_disk = os.path.join(here, 'tmp', url.replace('https://', ''))
            if not os.path.exists(file_on_disk):
                continue

            date_str = url.split('/')[-2]
            file_on_disk = rename_files_for_playwright(file_on_disk=file_on_disk)
            raw_html = open(file_on_disk).read()
            page.goto('file://' + file_on_disk, timeout=None)

            accept = processor.score_webpage(page, raw_html) if processor else True
            if accept:
                print(f'success! found good webpage. saving...')
                src = os.path.join(here, 'tmp', 'web.archive.org', 'web')
                dest = os.path.join(here, args.output_dir, 'web.archive.org', 'web')
                for dirname in list(filter(lambda x: date_str in x, os.listdir(src))):
                    src_dirname = os.path.join(src, dirname)
                    dest_dirname = os.path.join(dest, dirname)
                    shutil.move(src_dirname, dest_dirname)

                prev_datetime = url_seconds
            else:
                print('bad webpage, deleting...')
                to_delete = glob.glob(os.path.join(here, 'tmp', 'web.archive.org', 'web', date_str + '*'))
                for d in to_delete:
                    shutil.rmtree(d)







